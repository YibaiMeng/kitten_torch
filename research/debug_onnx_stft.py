"""
Trace the full forward STFT path and what feeds into noise_convs.
Also probe actual values for Div_1 (stage1÷2), stage2, decoder output, etc.
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

# Build output->node map
out_to_node = {}
for node in model_proto.graph.node:
    for out in node.output:
        out_to_node[out] = node

# Find noise_convs input
print("=== Tracing noise_convs inputs ===")
for node in model_proto.graph.node:
    if "noise_convs" in node.name:
        print(f"\n{node.name} ({node.op_type}):")
        print(f"  inputs: {list(node.input)}")
        for inp in node.input:
            if inp in out_to_node:
                pred = out_to_node[inp]
                print(f"  {inp!r} ← {pred.name} ({pred.op_type})")

# Trace the full sine→stft path
print("\n=== Forward STFT path (all nodes involving Conv, Conv_1 outputs) ===")
interesting = set()
for node in model_proto.graph.node:
    if node.name in ("/decoder/generator/Conv", "/decoder/generator/Conv_1"):
        interesting.update(node.output)

visited = set()
queue = list(interesting)
path_nodes = []
while queue:
    name = queue.pop(0)
    if name in visited:
        continue
    visited.add(name)
    if name in out_to_node:
        node = out_to_node[name]
        path_nodes.append(node)
        for out in node.output:
            queue.append(out)

for node in path_nodes:
    print(f"  {node.op_type}: {node.name!r}")
    print(f"    in:  {list(node.input)}")
    print(f"    out: {list(node.output)}")

# Probe actual ONNX values
print("\n=== Probing ONNX intermediate values ===")
voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

# Collect the cat/concat node that feeds noise_convs (the full STFT output)
# Also probe decoder output (features going into ups.0)
probe_names = []
for node in model_proto.graph.node:
    if "noise_convs" in node.name and node.op_type in ("ConvInteger", "Conv", "ConvTranspose"):
        probe_names.append(node.input[0])
    # ups.0 input = generator features from decoder
    if node.name == "/decoder/generator/ups.0/ConvTranspose" or "ups.0" in node.name:
        if node.op_type == "ConvTranspose":
            probe_names.append(node.input[0])
            break

# Get type map
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}
# float32=1, float16=10

probe_names = list(dict.fromkeys(probe_names))  # unique
print(f"Probing: {probe_names}")

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
            print(f"  range: {arr.min():.6f} to {arr.max():.6f}")
except Exception as e:
    print(f"Error running ONNX: {e}")
