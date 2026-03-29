"""Trace which style half feeds F0/N AdaIN blocks and text_encoder FCs."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")
import onnx
from onnx import numpy_helper

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
model_proto = onnx.load(ONNX_PATH)

out_to_node = {}
for node in model_proto.graph.node:
    for out in node.output:
        out_to_node[out] = node

init_map = {init.name: init for init in model_proto.graph.initializer}
graph_inputs = {inp.name for inp in model_proto.graph.input}

def get_slice_info(name, depth=0):
    """Trace back to find Slice of style input."""
    if depth > 15:
        return f"(depth limit)"
    if name in graph_inputs:
        return f"INPUT:{name}"
    if name in init_map:
        arr = numpy_helper.to_array(init_map[name])
        return f"INIT:{name}(shape={arr.shape})"
    if name not in out_to_node:
        return f"UNKNOWN:{name}"
    node = out_to_node[name]
    if node.op_type == "Slice":
        starts_name = node.input[1] if len(node.input) > 1 else None
        ends_name = node.input[2] if len(node.input) > 2 else None
        starts = ends = None
        if starts_name and starts_name in init_map:
            starts = list(numpy_helper.to_array(init_map[starts_name]))
        if ends_name and ends_name in init_map:
            ends = list(numpy_helper.to_array(init_map[ends_name]))
        src = get_slice_info(node.input[0], depth+1)
        return f"Slice[{starts}:{ends}] of ({src})"
    elif node.op_type in ("Cast", "Squeeze", "Unsqueeze", "Reshape", "Transpose",
                          "Flatten", "DynamicQuantizeLinear"):
        return get_slice_info(node.input[0], depth+1)
    elif node.op_type == "MatMulInteger":
        return get_slice_info(node.input[0], depth+1)
    elif node.op_type == "Gather":
        return f"Gather({get_slice_info(node.input[0], depth+1)})"
    return f"{node.op_type}:{node.name}"

# Check F0.0/norm1/fc MatMulInteger - input[0] is activations (style)
print("=== F0 AdaIN FCs ===")
for node in model_proto.graph.node:
    if node.op_type == 'MatMulInteger' and '/F0' in node.name and 'fc' in node.name:
        src = get_slice_info(node.input[0])
        print(f"  {node.name!r}")
        print(f"    activations -> {src}")

print("\n=== N AdaIN FCs ===")
for node in model_proto.graph.node:
    if node.op_type == 'MatMulInteger' and '/N.' in node.name and 'fc' in node.name:
        src = get_slice_info(node.input[0])
        print(f"  {node.name!r}")
        print(f"    activations -> {src}")

# Find text_encoder lstm FCs
print("\n=== text_encoder/lstms FCs ===")
for node in model_proto.graph.node:
    if node.op_type == 'MatMulInteger' and 'lstms' in node.name:
        src = get_slice_info(node.input[0])
        print(f"  {node.name!r}")
        print(f"    activations -> {src}")

# Also check F0/N Mul nodes (1/sqrt(2) scaling?)
print("\n=== F0/N Mul nodes (final) ===")
for node in model_proto.graph.node:
    if node.op_type == 'Mul' and (('/F0.' in node.name or '/N.' in node.name) and node.name.endswith('/Mul')):
        print(f"  {node.name!r}: inputs={list(node.input)}")
        for inp in node.input:
            if inp in init_map:
                arr = numpy_helper.to_array(init_map[inp])
                print(f"    init {inp!r}: {arr}")
