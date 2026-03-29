"""Check l_linear weight loading and sine scale."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import torch

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"

model_proto = onnx.load(ONNX_PATH)
init_dict = {init.name: np.array(onnx.numpy_helper.to_array(init)) for init in model_proto.graph.initializer}

# Check onnx::MatMul_8321
print("=== l_linear weight (onnx::MatMul_8321) ===")
if 'onnx::MatMul_8321' in init_dict:
    w = init_dict['onnx::MatMul_8321']
    print(f"  shape={w.shape}, range={w.min():.6f} to {w.max():.6f}")
    print(f"  values={w.flatten()}")

# Check all 'm_source' or 'l_linear' or 'l_sin' related initializers
print("\n=== All m_source initializers ===")
for name, arr in init_dict.items():
    if 'm_source' in name or 'l_linear' in name or 'sine' in name.lower() or 'l_sin' in name:
        print(f"  {name!r}: shape={arr.shape}, range={arr.min():.6f} to {arr.max():.6f}")

# Check PT model l_linear weight
from kitten_torch.model import build_model
model = build_model(ONNX_PATH)
model.eval()
gen = model.decoder.generator

print(f"\nPT l_linear.weight: {gen.sine_gen.l_linear.weight.shape}")
print(f"  range: {gen.sine_gen.l_linear.weight.min():.6f} to {gen.sine_gen.l_linear.weight.max():.6f}")
print(f"  values: {gen.sine_gen.l_linear.weight.data.flatten()}")
print(f"PT l_linear.bias: {gen.sine_gen.l_linear.bias.data.item():.6f}")

# Compare with onnx weight
if 'onnx::MatMul_8321' in init_dict:
    onnx_w = init_dict['onnx::MatMul_8321']
    pt_w = gen.sine_gen.l_linear.weight.data.numpy()
    print(f"\nONNX l_linear weight: {onnx_w.flatten()}")
    print(f"PT l_linear weight: {pt_w.flatten()}")
    print(f"Difference: {np.abs(onnx_w - pt_w.T).max():.6f}")  # may need transpose
    print(f"PT = ONNX.T?: {np.allclose(pt_w, onnx_w.T, atol=1e-4)}")

# Check our weight loader
print("\n=== Weight loader - how is l_linear loaded? ===")
from kitten_torch.weight_loader import ONNXWeights
wl = ONNXWeights(ONNX_PATH)
# Find any key with 'l_linear' or 'm_source'
print("Keys with 'l_linear' in weight loader:")
for key in sorted(wl.weights.keys()):
    if 'l_linear' in key or 'm_source' in key:
        arr = wl.weights[key]
        print(f"  {key!r}: shape={arr.shape}")
