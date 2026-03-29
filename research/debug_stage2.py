"""Probe ONNX stage 2 intermediates."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")
import numpy as np
import onnx
import onnxruntime as ort

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

voices = np.load(VOICES_PATH)
style_np = voices[list(voices.keys())[0]][0:1].astype(np.float32)
input_ids_np = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

model_proto = onnx.load(ONNX_PATH)
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}

# Find ONNX Add_3_output nodes in generator and locate stage2 intermediates
# Print all generator node names that are Add
print("=== Generator top-level Add/Div nodes ===")
for node in model_proto.graph.node:
    if node.name.startswith('/decoder/generator/') and '/' not in node.name[len('/decoder/generator/'):]:
        print(f"  {node.op_type}: {node.name!r} out={list(node.output)}")

print("\n=== Looking for resblocks.2 and .3 Add outputs ===")
for node in model_proto.graph.node:
    if node.op_type == 'Add' and '/decoder/generator/resblocks.2' in node.name and node.name.count('/') == 4:
        print(f"  {node.name!r}: out={list(node.output)}")
    if node.op_type == 'Add' and '/decoder/generator/resblocks.3' in node.name and node.name.count('/') == 4:
        print(f"  {node.name!r}: out={list(node.output)}")
