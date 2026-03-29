"""Debug Snake activation in resblocks."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
import onnx
from kitten_torch.model import build_model

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

voices = np.load(VOICES_PATH)
style_np = voices[list(voices.keys())[0]][0:1].astype(np.float32)

model = build_model(ONNX_PATH)
model.eval()
style = torch.from_numpy(style_np)
s = style[:, 128:]  # generator uses s2

# Get resblocks.0 (stage1, 128-ch) and resblocks.2 (stage2, 64-ch)
rb0 = model.decoder.generator.resblocks[0]
rb2 = model.decoder.generator.resblocks[2]

# Check alpha values
print("resblocks.0 alpha1[0] stats:")
print(f"  shape={rb0.alpha1[0].shape}, range={rb0.alpha1[0].min():.4f} to {rb0.alpha1[0].max():.4f}")
print(f"  mean={rb0.alpha1[0].mean():.4f}")
print("resblocks.2 alpha1[0] stats:")
print(f"  shape={rb2.alpha1[0].shape}, range={rb2.alpha1[0].min():.4f} to {rb2.alpha1[0].max():.4f}")
print(f"  mean={rb2.alpha1[0].mean():.4f}")

# Test Snake activation manually
x = torch.randn(1, 128, 100) * 10
alpha = rb0.alpha1[0]
print(f"\nTest snake (128-ch, |x|~10):")
print(f"  x: {x.min():.4f} to {x.max():.4f}")
snake_out = x + torch.sin(alpha * x).pow(2) / alpha
print(f"  snake(x): {snake_out.min():.4f} to {snake_out.max():.4f}")

# Test with large x (like from decoder: 113)
x_large = torch.ones(1, 128, 100) * 50  # simulate large input
snake_out_large = x_large + torch.sin(alpha * x_large).pow(2) / alpha
print(f"\nTest snake (x=50 constant):")
print(f"  snake(50): range={snake_out_large.min():.4f} to {snake_out_large.max():.4f}")

# Step through resblock.0 with x~35 to see where explosion might happen
print("\n=== Step-through resblocks.0 with x=35 ===")
x = torch.randn(1, 128, 540) * 10  # Stage 1 typical input
for i in range(3):
    h = rb0.adain1[i](x, s)
    print(f"  step {i} adain1: {h.min():.4f} to {h.max():.4f}")
    h_snake = h + torch.sin(rb0.alpha1[i] * h).pow(2) / rb0.alpha1[i]
    print(f"  step {i} snake1: {h_snake.min():.4f} to {h_snake.max():.4f}")
    h_conv1 = rb0.convs1[i](h_snake)
    print(f"  step {i} conv1: {h_conv1.min():.4f} to {h_conv1.max():.4f}")
    h2 = rb0.adain2[i](h_conv1, s)
    print(f"  step {i} adain2: {h2.min():.4f} to {h2.max():.4f}")
    h2_snake = h2 + torch.sin(rb0.alpha2[i] * h2).pow(2) / rb0.alpha2[i]
    print(f"  step {i} snake2: {h2_snake.min():.4f} to {h2_snake.max():.4f}")
    h2_conv2 = rb0.convs2[i](h2_snake)
    print(f"  step {i} conv2: {h2_conv2.min():.4f} to {h2_conv2.max():.4f}")
    x = x + h2_conv2
    print(f"  step {i} residual: {x.min():.4f} to {x.max():.4f}")
print(f"Final: {x.min():.4f} to {x.max():.4f}")
