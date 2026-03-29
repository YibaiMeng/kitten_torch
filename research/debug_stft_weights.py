"""Check if forward STFT weights are loaded correctly from ONNX."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import torch
from kitten_torch.model import build_model

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"

model_proto = onnx.load(ONNX_PATH)

# Find the forward STFT Conv nodes and their weight initializer names
print("=== Forward STFT Conv nodes ===")
stft_conv_weights = {}
for node in model_proto.graph.node:
    if node.name in ['/decoder/generator/Conv', '/decoder/generator/Conv_1']:
        print(f"  {node.name!r}: inputs={list(node.input)}")
        # node.input[1] is the weight, node.input[2] is the bias
        stft_conv_weights[node.name] = list(node.input)

# Get their weights from initializers
init_dict = {init.name: init for init in model_proto.graph.initializer}
for conv_name, inputs in stft_conv_weights.items():
    weight_name = inputs[1] if len(inputs) > 1 else None
    bias_name = inputs[2] if len(inputs) > 2 else None
    print(f"\n{conv_name}:")
    print(f"  weight_name: {weight_name!r}")
    if weight_name and weight_name in init_dict:
        w = np.array(onnx.numpy_helper.to_array(init_dict[weight_name]))
        print(f"  weight shape: {w.shape}, range: {w.min():.6f} to {w.max():.6f}")
    else:
        print(f"  weight NOT in initializers (may be a graph input or from another node)")
    if bias_name:
        print(f"  bias_name: {bias_name!r}")

# Check what name the weight_forward_real/imag would have
print("\n=== Looking for 'forward' or 'stft' in initializer names ===")
for init in model_proto.graph.initializer:
    if 'forward' in init.name.lower() or ('stft' in init.name.lower() and 'backward' not in init.name.lower()):
        arr = np.array(onnx.numpy_helper.to_array(init))
        print(f"  {init.name!r}: shape={arr.shape}, range={arr.min():.6f} to {arr.max():.6f}")

print("\n=== Looking for Conv in generator (non-quantized) ===")
for init in model_proto.graph.initializer:
    name = init.name
    if 'generator' in name and ('conv_post' in name.lower() or name.startswith('kmodel.decoder.decoder.generator.stft')
       or 'weight_forward' in name.lower() or 'weight_backward' in name.lower()):
        arr = np.array(onnx.numpy_helper.to_array(init))
        print(f"  {name!r}: shape={arr.shape}, range={arr.min():.6f} to {arr.max():.6f}")

# Check PT model stft weights
model = build_model(ONNX_PATH)
model.eval()
stft = model.decoder.generator.stft
print(f"\nPT weight_forward_real: {stft.weight_forward_real.shape}, range={stft.weight_forward_real.min():.6f} to {stft.weight_forward_real.max():.6f}")
print(f"PT weight_forward_imag: {stft.weight_forward_imag.shape}, range={stft.weight_forward_imag.min():.6f} to {stft.weight_forward_imag.max():.6f}")
print(f"PT weight_backward_real: {stft.weight_backward_real.shape}")
print(f"PT weight_backward_imag: {stft.weight_backward_imag.shape}")

# Simulate forward STFT manually with a unit sine
x = torch.sin(torch.linspace(0, 2*3.14159*440, 16200)).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    stft_out = stft.forward_stft(x)
print(f"\nTest forward_stft(440Hz sine): {stft_out.shape}")
print(f"  magnitude channels [0:11]: {stft_out[:,0:11,:].min():.4f} to {stft_out[:,0:11,:].max():.4f}")
print(f"  phase channels [11:22]: {stft_out[:,11:22,:].min():.4f} to {stft_out[:,11:22,:].max():.4f}")
