"""
Extract ONNX intermediate tensor values to compare with PyTorch.
Use ONNX node names to probe decoder output / generator input.
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

voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)

# Load ONNX model and get node names to find generator input
model_proto = onnx.load(ONNX_PATH)

# Find relevant output nodes (looking for generator/conv_post, ups output, etc.)
target_keywords = [
    "conv_post", "generator", "ups", "decode", "encode",
    "waveform", "istft", "backward"
]

# Collect value_info (intermediate tensor shapes)
print("=== Scanning ONNX graph for generator-related nodes ===")
# Get all node outputs and look for pattern
nodes_with_outputs = []
for node in model_proto.graph.node:
    for out in node.output:
        for kw in target_keywords:
            if kw.lower() in out.lower() or kw.lower() in node.name.lower():
                nodes_with_outputs.append((node.name, node.op_type, out))
                break

# Print unique node names containing target keywords
seen = set()
for name, op, out in nodes_with_outputs[:30]:
    if name not in seen:
        print(f"  node={name!r}, op={op}, output={out!r}")
        seen.add(name)

# Get specific intermediate outputs by adding them to the model
print("\n=== Adding intermediate outputs to ONNX for probing ===")

# Look for the last node before the waveform output
# Find all output names
graph_output_names = [o.name for o in model_proto.graph.output]
print(f"Graph outputs: {graph_output_names}")

# Find nodes producing these outputs
for node in model_proto.graph.node:
    for out in node.output:
        if out in graph_output_names:
            print(f"  Node producing {out!r}: name={node.name!r}, op={node.op_type}, inputs={list(node.input)[:3]}")

# Find Conv_post and the node before generator
# Scan for Conv nodes in the generator section
conv_nodes = []
for node in model_proto.graph.node:
    if node.op_type in ("Conv", "ConvTranspose", "Squeeze", "Cast"):
        for out in node.output:
            conv_nodes.append((node.name, node.op_type, list(node.input), out))

print(f"\nLast 20 Conv/ConvTranspose/Squeeze/Cast nodes:")
for name, op, inputs, out in conv_nodes[-20:]:
    print(f"  {op}: name={name!r}, out={out!r}")

# Extract intermediate by creating a modified model
print("\n=== Running ONNX with intermediate node output ===")

# Find the output of the conv_post node (Conv with bias coming just before iSTFT)
# We need to identify the right node - look for Conv1d producing (B, 22, T) output
# Strategy: add value_info outputs for key nodes and run

# Find all intermediate value_info names
all_vi = {vi.name: vi for vi in model_proto.graph.value_info}

# Find nodes that output tensors that could be conv_post output
# The last Conv node before the inverse STFT
all_outputs_order = []
for node in model_proto.graph.node:
    for out in node.output:
        all_outputs_order.append((node.op_type, node.name, out))

# Print the last 50 ops
print("\nLast 50 ops in graph:")
for op, name, out in all_outputs_order[-50:]:
    print(f"  {op}: out={out!r}")
