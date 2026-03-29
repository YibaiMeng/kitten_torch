"""
Compare key weights between ONNX and PT model to find discrepancies.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import onnx
from onnx import numpy_helper

from kitten_torch.model import build_model
from kitten_torch.weight_loader import ONNXWeights

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"

print("Loading model...")
model = build_model(ONNX_PATH)
model.eval()

w = ONNXWeights(ONNX_PATH)

print("\n=== Generator weight checks ===")

gp = "kmodel.decoder.decoder.generator"

# Check conv_post weights (float16)
pt_conv_post_w = model.decoder.generator.conv_post.weight.data
onnx_conv_post_w = torch.from_numpy(w.raw(f"{gp}.conv_post.weight").astype(np.float32))
print(f"conv_post weight match: {torch.allclose(pt_conv_post_w, onnx_conv_post_w, atol=1e-4)}")
print(f"  PT range: {pt_conv_post_w.min():.4f} to {pt_conv_post_w.max():.4f}")
print(f"  ONNX range: {onnx_conv_post_w.min():.4f} to {onnx_conv_post_w.max():.4f}")

# Check ups[0] weights
pt_ups0_w = model.decoder.generator.ups[0].weight.data
onnx_ups0_w = torch.from_numpy(w.raw(f"{gp}.ups.0.weight").astype(np.float32))
print(f"\nups.0 weight shape: PT={pt_ups0_w.shape}, ONNX={onnx_ups0_w.shape}")
print(f"ups.0 weight match: {torch.allclose(pt_ups0_w, onnx_ups0_w, atol=1e-4)}")

# Check STFT weights
print("\n=== STFT weight checks ===")
for attr in ["weight_forward_real", "weight_forward_imag", "weight_backward_real", "weight_backward_imag"]:
    pt_w = getattr(model.decoder.generator.stft, attr).data
    onnx_w = torch.from_numpy(w.raw(f"{gp}.stft.{attr}").astype(np.float32))
    print(f"{attr}: PT shape={pt_w.shape}, ONNX shape={onnx_w.shape}")
    print(f"  match: {torch.allclose(pt_w, onnx_w, atol=1e-4)}")
    print(f"  PT range: {pt_w.min():.4f} to {pt_w.max():.4f}")
    print(f"  ONNX range: {onnx_w.min():.4f} to {onnx_w.max():.4f}")

# Check AdaIN fc weights for a resblock
print("\n=== Generator resblock[0] adain1[0] fc check ===")
prefix = f"{gp}.resblocks.0.adain1.0"
pt_fc_w = model.decoder.generator.resblocks[0].adain1[0].fc.weight.data
onnx_fc_w_raw = w.get(f"{prefix}.fc.weight_quantized")  # dequantized
print(f"adain1.0.fc: PT shape={pt_fc_w.shape}, ONNX raw shape={onnx_fc_w_raw.shape}")
# PT weight is (2*channels, style_half) = (256, 128)
# ONNX weight is stored as (128, 256) → transposed = (256, 128)
print(f"  PT fc.weight range: {pt_fc_w.min():.4f} to {pt_fc_w.max():.4f}")
print(f"  ONNX fc.weight (dequant) range: {onnx_fc_w_raw.min():.4f} to {onnx_fc_w_raw.max():.4f}")
print(f"  Match after transpose: {torch.allclose(pt_fc_w, onnx_fc_w_raw.T, atol=0.01)}")

# Check adain norm weights
onnx_norm_w = w.get(f"{prefix}.norm.weight")
onnx_norm_b = w.get(f"{prefix}.norm.bias")
pt_norm_w = model.decoder.generator.resblocks[0].adain1[0].norm.weight.data
pt_norm_b = model.decoder.generator.resblocks[0].adain1[0].norm.bias.data
print(f"\nadain1.0.norm weight: {torch.allclose(pt_norm_w, onnx_norm_w, atol=1e-4)}")
print(f"adain1.0.norm bias:   {torch.allclose(pt_norm_b, onnx_norm_b, atol=1e-4)}")

# Test iSTFT manually vs ONNX approach
print("\n=== iSTFT equivalence test ===")
import torch.nn.functional as F

stft = model.decoder.generator.stft
# Create synthetic spec
torch.manual_seed(42)
spec = torch.randn(1, 22, 100)  # (B, 22, frames)

# Our loop approach
with torch.no_grad():
    out_loop = stft.inverse_stft(spec)

    # ONNX approach: single ConvTranspose call
    # weight_backward_real: (11, 1, 20) — interprets as 11-in, 1-out
    real = spec[:, :11, :]
    imag = spec[:, 11:, :]
    out_real = F.conv_transpose1d(real, stft.weight_backward_real, stride=5)  # (B, 1, N)
    out_imag = F.conv_transpose1d(imag, stft.weight_backward_imag, stride=5)  # (B, 1, N)
    out_onnx = out_real - out_imag

print(f"Loop: shape={out_loop.shape}, range={out_loop.min():.4f} to {out_loop.max():.4f}")
print(f"ONNX: shape={out_onnx.shape}, range={out_onnx.min():.4f} to {out_onnx.max():.4f}")
print(f"Match: {torch.allclose(out_loop, out_onnx[:, :, :out_loop.shape[-1]], atol=1e-4)}")
print(f"Max diff: {(out_loop - out_onnx[:, :, :out_loop.shape[-1]]).abs().max():.6f}")
