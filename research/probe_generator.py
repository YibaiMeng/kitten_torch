"""
Probe generator intermediates: compare PT vs ONNX after MRF blocks.
Focus: MRF averaging (is ONNX * 0.5 or just adding)?
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import re

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

TEXT = "Hello, how are you today?"
VOICE = "expr-voice-2-m"

from kitten_torch import KittenTTS
tts = KittenTTS(model_path=ONNX_PATH, voices_path=VOICES_PATH)
model = tts._model
model.eval()

voices_data = np.load(VOICES_PATH)
style_np = voices_data[VOICE][0:1].astype(np.float32)
style_pt = torch.from_numpy(style_np)
ids = tts._phonemize(TEXT)
input_ids = torch.tensor([ids], dtype=torch.long)

# PT hooks
captured = {}
def make_hook(name):
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured[name] = o.detach().clone()
    return hook

gen = model.decoder.generator
# Hook on individual resblocks
for i, rb in enumerate(gen.resblocks):
    rb.register_forward_hook(make_hook(f"resblock_{i}"))
gen.ups[0].register_forward_hook(make_hook("ups0"))
gen.ups[1].register_forward_hook(make_hook("ups1"))
gen.noise_res[0].register_forward_hook(make_hook("noise_res0"))
gen.noise_res[1].register_forward_hook(make_hook("noise_res1"))
gen.conv_post.register_forward_hook(make_hook("conv_post"))

with torch.inference_mode():
    audio_pt = model(input_ids, style_pt, deterministic=True)

print("=== PT generator intermediates ===")
for k, v in captured.items():
    v = v.float()
    print(f"  {k}: shape={tuple(v.shape)}  mean={v.mean():.4f}  std={v.std():.4f}  max={v.abs().max():.4f}")
print(f"  audio: max={audio_pt.abs().max():.4f}")

# MRF check in PT: resblock_0 + resblock_1 vs ups0 + noise
r0 = captured['resblock_0'].float()
r1 = captured['resblock_1'].float()
print(f"\n  r0+r1 stats: mean={(r0+r1).mean():.4f}  std={(r0+r1).std():.4f}  max={(r0+r1).abs().max():.4f}")
print(f"  (r0+r1)*0.5: mean={(0.5*(r0+r1)).mean():.4f}  std={(0.5*(r0+r1)).std():.4f}  max={(0.5*(r0+r1)).abs().max():.4f}")

# ONNX probe: add resblocks and generator-level Mul as outputs
print("\n=== Probing ONNX generator ===")
model_onnx = onnx.load(ONNX_PATH)

# Find all relevant generator node outputs
target_names = []
for node in model_onnx.graph.node:
    for out in node.output:
        if "/decoder/generator/" in out:
            target_names.append(out)

# Deduplicate, sort
seen = set()
unique = []
for n in target_names:
    if n not in seen:
        seen.add(n)
        unique.append(n)

print("Generator ONNX nodes (first 60):")
for n in unique[:60]:
    print(f"  {n}")

# Find specific nodes: resblocks output Mul, and the top-level MRF combination
# Look for Add nodes after resblocks
add_after_resblock = []
for node in model_onnx.graph.node:
    if node.op_type == "Add":
        for inp in node.input:
            if "resblocks" in inp:
                for out in node.output:
                    add_after_resblock.append((inp, out))

print(f"\nAdd nodes consuming resblock outputs:")
for inp, out in add_after_resblock[:10]:
    print(f"  {inp} → {out}")

# Find Mul nodes after Add of resblocks (the * 0.5)
# First find names of Add outputs
add_outputs = {out for inp, out in add_after_resblock}

mul_after_add = []
for node in model_onnx.graph.node:
    if node.op_type == "Mul":
        for inp in node.input:
            if inp in add_outputs:
                for out in node.output:
                    mul_after_add.append((inp, out))

print(f"\nMul nodes after Add-of-resblocks (MRF average?):")
for inp, out in mul_after_add[:10]:
    print(f"  {inp} → {out}")

# Now extract the names we want to probe
probe_names = []
# generator/Mul_output_0 - the Mul in ONNX
for n in unique:
    if "/decoder/generator/Mul_output_0" in n:
        probe_names.append(n)
    if "resblocks" in n and "Mul_output_0" in n:
        probe_names.append(n)

print(f"\nProbing ONNX nodes: {probe_names}")

# Add as outputs
for pname in probe_names:
    vi = helper.make_tensor_value_info(pname, TensorProto.FLOAT, None)
    model_onnx.graph.output.append(vi)

# Also find the Add-of-resblocks output
for inp, out in add_after_resblock:
    if "resblocks.0" in inp or "resblocks.1" in inp:
        vi = helper.make_tensor_value_info(out, TensorProto.FLOAT, None)
        model_onnx.graph.output.append(vi)
        probe_names.append(out)
        break  # just the first one

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

    onnx_audio = outputs[0].astype(np.float32)
    print(f"\nONNX audio max: {np.abs(onnx_audio).max():.4f}")

    n_base = 1  # skip the first (main audio output)
    print("\n=== ONNX generator probes ===")
    for i, pname in enumerate(probe_names):
        val = outputs[n_base + i].astype(np.float32)
        print(f"  {pname.split('/decoder/generator/')[-1]}: shape={val.shape}  mean={val.mean():.4f}  std={val.std():.4f}  max={np.abs(val).max():.4f}")

except Exception as e:
    print(f"ONNX probe error: {e}")
    import traceback
    traceback.print_exc()

    # Retry with specific float16 issue
    print("\nLooking for which nodes are float16...")
    for pname in probe_names:
        for vi in model_onnx.graph.value_info:
            if vi.name == pname:
                print(f"  {pname}: type={vi.type.tensor_type.elem_type}")
