"""
Probe ONNX decoder output and compare with our PT decoder output.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
import torch
import torch.nn.functional as F

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

model_proto = onnx.load(ONNX_PATH)

# Track what feeds into the generator's LeakyReLU (the generator input)
# We know: "/decoder/generator/LeakyRelu_output_0" = generator input after LeakyReLU
# "/decoder/generator/LeakyRelu_output_0_Cast_to_float16_input_0" should be the pre-cast float32 version

# Find the decode.3 output, encode output etc.
print("=== Searching for decoder output nodes ===")
decode_outputs = []
for node in model_proto.graph.node:
    if "decode" in node.name and "generator" not in node.name:
        if node.op_type in ("Add", "Cast"):
            for out in node.output:
                decode_outputs.append((node.name, out))

# Print the last few decode outputs
for name, out in decode_outputs[-20:]:
    print(f"  {name!r}: out={out!r}")

# Find what node produces the LeakyRelu input
print("\n=== Finding LeakyRelu input node ===")
leaky_out = "/decoder/generator/LeakyRelu_output_0_Cast_to_float16_input_0"
for node in model_proto.graph.node:
    for out in node.output:
        if out == leaky_out:
            print(f"Produced by: {node.name!r} ({node.op_type})")
            print(f"  inputs: {list(node.input)}")

# vi_types
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}

# Probe key values
voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

probe_names = [
    "/decoder/generator/LeakyRelu_output_0_Cast_to_float16_input_0",  # decoder output (float32)
    "/decoder/generator/Div_1_output_0",    # stage 1 ÷2 (float32 from Cast)
    "/decoder/generator/Div_2_output_0",    # stage 2 ÷2 (float32 from Cast)
]

# Also try to find the decode.3 output before going to generator
# Look for Cast nodes near the generator input
for node in model_proto.graph.node:
    if node.name == "/decoder/generator/LeakyRelu_output_0_Cast_to_float16_0":
        print(f"\nLeakyRelu float16 cast node:")
        print(f"  inputs: {list(node.input)}")
        print(f"  outputs: {list(node.output)}")
        probe_names.insert(0, node.input[0])  # The float32 input to this cast

model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)
for name in probe_names:
    elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
    vi = onnx.helper.make_tensor_value_info(name, elem_type, None)
    model_copy.graph.output.append(vi)

try:
    sess = ort.InferenceSession(model_copy.SerializeToString(), providers=["CPUExecutionProvider"])
    results = sess.run(None, {"input_ids": input_ids, "style": style_np, "speed": speed})
    output_names = [o.name for o in sess.get_outputs()]
    probe_set = set(probe_names)
    for name, result in zip(output_names, results):
        if name in probe_set:
            arr = np.asarray(result, dtype=np.float32)
            print(f"\n{name!r}:")
            print(f"  shape={arr.shape}")
            print(f"  range: {arr.min():.6f} to {arr.max():.6f}")
            print(f"  abs mean: {np.abs(arr).mean():.4f}")
except Exception as e:
    print(f"Error: {e}")

# Compare with PT
print("\n=== PT decoder output ===")
from kitten_torch.model import build_model
model = build_model(ONNX_PATH)
model.eval()

style = torch.from_numpy(style_np)
ids = torch.from_numpy(input_ids)

with torch.no_grad():
    text_proj, text_lstm = model.text_encoder(ids)
    bert_out = model.bert(ids)
    text_enc_out, durations = model.predictor.forward(bert_out, style, 1.0)
    dur_int = durations.round().long().clamp(min=1)
    lr_features = model._length_regulate(text_enc_out, dur_int)
    shared_h, f0_pred, n_pred = model.predictor.forward_frame(lr_features, style)
    text_proj_lr = model._length_regulate_proj(text_proj, dur_int)
    features, f0_2T = model.decoder(text_proj_lr, f0_pred, n_pred, style)

print(f"PT decoder features: shape={features.shape}, range={features.min():.4f} to {features.max():.4f}")
print(f"PT after LeakyReLU: range={F.leaky_relu(features, 0.2).min():.4f} to {F.leaky_relu(features, 0.2).max():.4f}")
