"""
Find ONNX Slice constant values and probe iSTFT using a minimal ONNX.
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

# Build a map of constant output→value
const_map = {}
for node in model_proto.graph.node:
    if node.op_type == "Constant":
        for out in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    const_map[out] = numpy_helper.to_array(attr.t)

print("=== Slice_1 constants ===")
slice1_inputs = None
slice2_inputs = None
for node in model_proto.graph.node:
    if node.name == "/decoder/generator/Slice_1":
        slice1_inputs = list(node.input)
        print(f"Slice_1 inputs: {slice1_inputs}")
        for i, inp in enumerate(slice1_inputs):
            if inp in const_map:
                print(f"  {inp}: {const_map[inp]}")
    if node.name == "/decoder/generator/Slice_2":
        slice2_inputs = list(node.input)
        print(f"Slice_2 inputs: {slice2_inputs}")
        for i, inp in enumerate(slice2_inputs):
            if inp in const_map:
                print(f"  {inp}: {const_map[inp]}")

# Also check the forward STFT's Div
print("\n=== Forward STFT Div info ===")
for node in model_proto.graph.node:
    if node.name == "/decoder/generator/Div":
        print(f"Div: {list(node.input)} → {list(node.output)}")
        # Is Div_output used as input to noise_convs?
        div_output = node.output[0]
        print(f"  Looking for users of {div_output}...")
        for n in model_proto.graph.node:
            if div_output in n.input:
                print(f"  Used by: {n.name} ({n.op_type})")

print("\n=== Probing actual values via ONNX (for conv_post output) ===")
voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

# Use float16 outputs with UNDEFINED → need proper type
# Let me probe with float32 cast
# Try probing the LeakyRelu_2 output (float16 internal) - it feeds conv_post
# conv_post inputs: LeakyRelu_2 (float16), weight, bias → output float16 → cast float32

# Try just adding the Cast output (which is float32)
model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)

# Only add the float32 outputs (after Cast nodes)
float32_probes = [
    "/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_output_0",
    "/decoder/generator/Div_1_output_0",
    "/decoder/generator/Div_2_output_0",
]

# Check value_info types
vi_types = {}
for vi in model_proto.graph.value_info:
    vi_types[vi.name] = vi.type

# Add each probe as float32 output
for name in float32_probes:
    if name in vi_types:
        vi = onnx.helper.make_tensor_value_info(
            name,
            vi_types[name].tensor_type.elem_type,
            None
        )
    else:
        # Try float32
        vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
    model_copy.graph.output.append(vi)

try:
    sess = ort.InferenceSession(model_copy.SerializeToString(), providers=["CPUExecutionProvider"])
    results = sess.run(None, {
        "input_ids": input_ids,
        "style": style_np,
        "speed": speed,
    })
    output_names = [o.name for o in sess.get_outputs()]
    probe_set = set(float32_probes)
    for name, result in zip(output_names, results):
        if name in probe_set:
            arr = np.asarray(result, dtype=np.float32)
            print(f"\n{name!r}:")
            print(f"  shape={arr.shape}")
            print(f"  range: {arr.min():.6f} to {arr.max():.6f}")
except Exception as e:
    print(f"Error: {e}")

# Also, let's understand the Div_output_0 = Conv_1/Conv  (imag/real of forward STFT)
print("\n=== Understanding forward STFT Div ===")
# The Div outputs are used in the noise path:
# Conv (sine, forward_real) → real_spec
# Conv_1 (sine, forward_imag) → imag_spec
# Div = imag_spec / real_spec = tan(phase)
# But what is this used for?
print("Forward STFT: Div = imag/real (phase tangent?)")
for node in model_proto.graph.node:
    if node.name == "/decoder/generator/Div":
        div_out = node.output[0]
        users = [n.name for n in model_proto.graph.node if div_out in n.input]
        print(f"  Div output users: {users}")

# Find what follows Conv and Conv_1 in the noise path
print("\nNodes using forward STFT outputs:")
for conv_name in ["/decoder/generator/Conv", "/decoder/generator/Conv_1"]:
    for node in model_proto.graph.node:
        if node.name == conv_name:
            conv_out = node.output[0]
            users = [(n.name, n.op_type) for n in model_proto.graph.node if conv_out in n.input]
            print(f"  {conv_name} → {conv_out} used by: {users}")
