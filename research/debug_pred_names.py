"""Find predictor-related node names in ONNX."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")
import onnx

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
model_proto = onnx.load(ONNX_PATH)

# Print all unique name prefixes in predictor nodes
predictor_nodes = [n for n in model_proto.graph.node if 'predictor' in n.name.lower() or 'Predictor' in n.name]
print(f"Total predictor nodes: {len(predictor_nodes)}")

# Print all Gemm/MatMul/MatMulInteger nodes with 'predictor'
print("\n=== Gemm/MatMul in predictor ===")
for node in model_proto.graph.node:
    if node.op_type in ('Gemm', 'MatMul', 'MatMulInteger') and 'predictor' in node.name:
        print(f"  {node.op_type}: {node.name!r}")

# Also check all node names containing 'F0' or 'f0'
print("\n=== F0 nodes ===")
for node in model_proto.graph.node:
    if 'F0' in node.name or '/f0' in node.name.lower():
        print(f"  {node.op_type}: {node.name!r}")

# Print first 20 predictor node names to understand the naming convention
print("\n=== First 30 predictor nodes ===")
for node in predictor_nodes[:30]:
    print(f"  {node.op_type}: {node.name!r}")
