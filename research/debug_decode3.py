"""
Trace what produces decode.3/Mul_output_0 (the generator input).
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

model_proto = onnx.load(ONNX_PATH)

# Find the node that produces decode.3/Mul_output_0
print("=== decode.3 section ===")
for node in model_proto.graph.node:
    if "/decoder/decode.3" in node.name:
        print(f"  {node.op_type}: {node.name!r}")
        print(f"    in:  {list(node.input)}")
        print(f"    out: {list(node.output)}")

# Build output->node map
out_to_node = {}
for node in model_proto.graph.node:
    for out in node.output:
        out_to_node[out] = node

# Trace from decode.3/Mul_output_0 backwards
target = "/decoder/decode.3/Mul_output_0"
print(f"\n=== Tracing backwards from {target!r} ===")

def trace_back(name, depth=0, max_depth=5):
    if depth > max_depth:
        return
    if name not in out_to_node:
        print(f"{'  '*depth}{name!r} (initializer or input)")
        return
    node = out_to_node[name]
    print(f"{'  '*depth}{node.op_type}: {node.name!r} → {name!r}")
    print(f"{'  '*depth}  inputs: {list(node.input)}")
    for inp in node.input[:3]:
        trace_back(inp, depth+1, max_depth)

trace_back(target)

# Also probe the actual value of decode.3/Mul_output_0
print("\n=== Probing decode.3/Mul_output_0 value ===")
voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}

probe_names = [
    "/decoder/decode.3/Mul_output_0",
    "/decoder/decode.3/Add_output_0",  # residual sum before Mul
]

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
            print(f"\n{name!r}: shape={arr.shape}")
            print(f"  range: {arr.min():.4f} to {arr.max():.4f}")
            print(f"  abs mean: {np.abs(arr).mean():.4f}")
except Exception as e:
    print(f"Error: {e}")
