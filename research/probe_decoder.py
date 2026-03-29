"""
Probe decoder intermediate values: PT vs ONNX.
Compares features at the decoder output (decode.3 output → generator input).
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnx import numpy_helper, TensorProto

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

TEXT = "Hello, how are you today?"
VOICE = "expr-voice-2-m"

# ── Build PT model ─────────────────────────────────────────────────────────────
from kitten_torch import KittenTTS
from kitten_torch.model import KittenTTSTorch, build_model
import re

tts = KittenTTS(model_path=ONNX_PATH, voices_path=VOICES_PATH)
model = tts._model
model.eval()

voices_data = np.load(VOICES_PATH)
style_np = voices_data[VOICE][0:1].astype(np.float32)  # (1, 256)
style_pt = torch.from_numpy(style_np)

ids = tts._phonemize(TEXT)
input_ids = torch.tensor([ids], dtype=torch.long)

# ── PT forward with intermediate captures ─────────────────────────────────────
captured = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            captured[name] = output[0].detach().clone()
        else:
            captured[name] = output.detach().clone()
    return hook

# Register hooks on decoder components
model.decoder.encode.register_forward_hook(make_hook("encode_out"))
for i, blk in enumerate(model.decoder.decode):
    blk.register_forward_hook(make_hook(f"decode_{i}_out"))
model.decoder.asr_res.register_forward_hook(make_hook("asr_res_out"))
model.decoder.F0_conv.register_forward_hook(make_hook("F0_conv_out"))
model.decoder.N_conv.register_forward_hook(make_hook("N_conv_out"))

# Also capture what enters the generator
model.decoder.generator.register_forward_hook(make_hook("generator_out"))

# Capture features entering generator's first ups
model.decoder.generator.ups[0].register_forward_hook(make_hook("ups0_out"))

with torch.inference_mode():
    audio_pt = model(input_ids, style_pt)

print("=== PT intermediate shapes ===")
for k, v in captured.items():
    print(f"  {k}: shape={tuple(v.shape)}  mean={v.float().mean().item():.4f}  std={v.float().std().item():.4f}  max={v.float().abs().max().item():.4f}")

print(f"\n  audio output: mean={audio_pt.float().mean().item():.4f}  max={audio_pt.float().abs().max().item():.4f}")

# ── ONNX: add intermediate outputs ────────────────────────────────────────────
print("\n=== Probing ONNX graph ===")
model_onnx = onnx.load(ONNX_PATH)

# Find relevant intermediate node names
target_patterns = [
    "decode.0",
    "decode.1",
    "decode.2",
    "decode.3",
    "encode",
    "generator",
    "ups.0",
    "F0_conv",
    "N_conv",
]

# List all intermediate value names that match our patterns
intermediate_names = []
for node in model_onnx.graph.node:
    for out in node.output:
        for pat in target_patterns:
            if pat in out and "Mul_output_0" in out:
                intermediate_names.append(out)
                break

# Also look for the features just before generator
for node in model_onnx.graph.node:
    for out in node.output:
        if "generator" in out and ("input" in out or "Cast" in out or "LeakyRelu" in out):
            intermediate_names.append(out)

# Print first 30 unique matching node outputs
seen = set()
print("Candidate intermediate node outputs:")
for name in intermediate_names:
    if name not in seen:
        seen.add(name)
        print(f"  {name}")

# Find the specific names for decode block outputs
# ONNX names pattern: /decoder/decoder/decode.X/Mul_output_0
decode_output_names = []
for node in model_onnx.graph.node:
    for out in node.output:
        if "/decoder/decode." in out and "Mul_output_0" in out:
            decode_output_names.append(out)
        elif "/decoder/decoder/decode." in out and "Mul_output_0" in out:
            decode_output_names.append(out)

print(f"\ndecode block Mul outputs: {decode_output_names}")

# Also find encode block output
encode_output_names = []
for node in model_onnx.graph.node:
    for out in node.output:
        if "encode" in out and "Mul_output_0" in out and "decode" not in out:
            encode_output_names.append(out)
print(f"encode block Mul outputs: {encode_output_names}")

# Find any output just before generator input (features)
gen_input_names = []
for node in model_onnx.graph.node:
    for out in node.output:
        if "Mul_output_0" in out and "decode" in out:
            gen_input_names.append(out)
gen_input_names = sorted(set(gen_input_names))
print(f"\nAll decode Mul outputs: {gen_input_names}")

# Now build ONNX session with the last decode output as intermediate
# Find the actual tensor name that goes INTO the generator
# Look at what generator's first ups.0 node consumes
ups0_input = None
for node in model_onnx.graph.node:
    if "ups.0" in node.name or (node.op_type == "ConvTranspose" and "ups" in str(node.input)):
        print(f"\nFound ups.0 node: {node.name}")
        print(f"  Inputs: {list(node.input)}")
        ups0_input = node.input[0] if node.input else None
        break

# Also check for the Cast → ConvTranspose path (float16 ups)
for node in model_onnx.graph.node:
    if node.op_type == "Cast":
        for out in node.output:
            if "Cast_to_float16" in out or "float16" in out:
                pass  # not printing all

# Find generator's first real computation
# The generator takes features from decoder.decode.3 output
# Let's find what node consumes the decode.3/Mul_output_0
last_decode_out = None
for node in model_onnx.graph.node:
    for out in node.output:
        if "decode.3" in out and "Mul_output_0" in out:
            last_decode_out = out

print(f"\nLast decode.3 Mul output name: {last_decode_out}")

if last_decode_out:
    # Add it as a graph output and run
    from onnx import helper
    new_output = helper.make_tensor_value_info(last_decode_out, TensorProto.FLOAT, None)
    model_onnx.graph.output.append(new_output)

    try:
        sess = ort.InferenceSession(
            model_onnx.SerializeToString(),
            providers=["CPUExecutionProvider"]
        )

        import kittentts as _kt, inspect
        src = inspect.getsource(_kt.KittenTTS.__init__)
        m = re.search(r"enumerate\(list\('(.+?)'\)\)", src, re.DOTALL)
        WORD_INDEX = {s: i for i, s in enumerate(list(m.group(1)))}

        phonemized = tts._phonemizer.phonemize([TEXT])[0]
        normalized = ' '.join(re.findall(r"\w+|[^\w\s]", phonemized))
        onnx_ids = [0]
        for c in normalized:
            if c in WORD_INDEX:
                onnx_ids.append(WORD_INDEX[c])
        onnx_ids.append(0)

        input_ids_np = np.array([onnx_ids], dtype=np.int64)
        speed_np = np.array([1.0], dtype=np.float32)

        outputs = sess.run(None, {
            "input_ids": input_ids_np,
            "style": style_np,
            "speed": speed_np,
        })

        # Last output is our intermediate
        onnx_audio = outputs[0]
        onnx_decode3 = outputs[-1]  # the decode.3 output

        print(f"\n=== ONNX decode.3 output ===")
        print(f"  shape: {onnx_decode3.shape}")
        d3f = onnx_decode3.astype(np.float32)
        print(f"  mean={d3f.mean():.4f}  std={d3f.std():.4f}  max={np.abs(d3f).max():.4f}")

        print(f"\n=== PT decode.3 output ===")
        pt_d3 = captured.get("decode_3_out")
        if pt_d3 is not None:
            print(f"  shape: {tuple(pt_d3.shape)}")
            print(f"  mean={pt_d3.float().mean().item():.4f}  std={pt_d3.float().std().item():.4f}  max={pt_d3.float().abs().max().item():.4f}")

            # Trim to matching time length
            T_onnx = onnx_decode3.shape[-1]
            T_pt = pt_d3.shape[-1]
            T = min(T_onnx, T_pt)
            diff = np.abs(onnx_decode3[..., :T].astype(np.float32) - pt_d3.numpy()[..., :T])
            print(f"\n  diff max={diff.max():.4f}  diff mean={diff.mean():.4f}")
            print(f"  correlation: {np.corrcoef(onnx_decode3[..., :T].flatten(), pt_d3.numpy()[..., :T].flatten())[0,1]:.4f}")

        print(f"\nONNX audio max: {np.abs(onnx_audio).max():.4f}")

    except Exception as e:
        print(f"\nONNX probe failed: {e}")
        import traceback
        traceback.print_exc()

        # Try with FLOAT16 type instead
        print("\nRetrying with FLOAT16 type annotation...")
        model_onnx2 = onnx.load(ONNX_PATH)
        new_output2 = helper.make_tensor_value_info(last_decode_out, TensorProto.FLOAT16, None)
        model_onnx2.graph.output.append(new_output2)
        try:
            sess2 = ort.InferenceSession(
                model_onnx2.SerializeToString(),
                providers=["CPUExecutionProvider"]
            )
            outputs2 = sess2.run(None, {
                "input_ids": input_ids_np,
                "style": style_np,
                "speed": speed_np,
            })
            onnx_decode3_fp16 = outputs2[-1].astype(np.float32)
            print(f"  ONNX decode.3 (fp16): shape={onnx_decode3_fp16.shape}  mean={onnx_decode3_fp16.mean():.4f}  std={onnx_decode3_fp16.std():.4f}  max={np.abs(onnx_decode3_fp16).max():.4f}")
        except Exception as e2:
            print(f"  fp16 retry failed: {e2}")
