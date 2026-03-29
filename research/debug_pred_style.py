"""
Trace which style half ([:128] or [128:]) feeds the predictor F0/N AdaIN blocks
and the text_encoder FCs in the ONNX graph.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"

model_proto = onnx.load(ONNX_PATH)

# Build output→node map and initializer set
out_to_node = {}
for node in model_proto.graph.node:
    for out in node.output:
        out_to_node[out] = node

init_names = {init.name for init in model_proto.graph.initializer}
graph_inputs = {inp.name for inp in model_proto.graph.input}

def get_slice_range(name, depth=0):
    """Trace back to find if a tensor comes from a Slice of 'style' input."""
    if depth > 10:
        return None
    if name in init_names or name in graph_inputs:
        return f"INPUT:{name}"
    if name not in out_to_node:
        return None
    node = out_to_node[name]
    if node.op_type == "Slice":
        # Try to find the start/end from initializers
        starts_name = node.input[1] if len(node.input) > 1 else None
        ends_name = node.input[2] if len(node.input) > 2 else None
        starts = ends = None
        for init in model_proto.graph.initializer:
            if init.name == starts_name:
                starts = list(onnx.numpy_helper.to_array(init))
            if init.name == ends_name:
                ends = list(onnx.numpy_helper.to_array(init))
        data_src = get_slice_range(node.input[0], depth+1)
        return f"Slice({data_src}, starts={starts}, ends={ends})"
    elif node.op_type in ("Cast", "Squeeze", "Unsqueeze", "Reshape", "Transpose", "Flatten"):
        return get_slice_range(node.input[0], depth+1)
    elif node.op_type == "MatMulInteger":
        # This is a quantized matmul — trace first input (activations)
        return get_slice_range(node.input[0], depth+1)
    elif node.op_type == "Gather":
        return get_slice_range(node.input[0], depth+1)
    elif node.op_type in ("DynamicQuantizeLinear",):
        return get_slice_range(node.input[0], depth+1)
    return f"{node.op_type}:{node.name}"

# Find all Gemm/MatMul nodes in predictor/F0, predictor/N, predictor/text_encoder
print("=== predictor F0 AdaIN FCs ===")
for node in model_proto.graph.node:
    name = node.name
    if ('/predictor/F0' in name or '/predictor/N' in name) and ('Gemm' in name or 'MatMul' in name):
        # Find which input is the style (not weight, not bias)
        # For quantized Gemm: inputs are [A(activations), B(weight), bias, ...]
        print(f"\n  {node.op_type}: {name!r}")
        for j, inp in enumerate(node.input[:3]):
            src = get_slice_range(inp)
            print(f"    input[{j}] {inp!r} -> {src}")

print("\n=== predictor/text_encoder AdaIN FCs ===")
for node in model_proto.graph.node:
    name = node.name
    if '/predictor/text_encoder' in name and ('Gemm' in name or 'MatMul' in name):
        print(f"\n  {node.op_type}: {name!r}")
        for j, inp in enumerate(node.input[:3]):
            src = get_slice_range(inp)
            print(f"    input[{j}] {inp!r} -> {src}")
