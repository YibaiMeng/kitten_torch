"""
Trace amplitude divergence between ONNX and PyTorch.
Compare generator intermediate outputs stage by stage.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort

from kitten_torch.model import build_model
from kitten_torch.weight_loader import ONNXWeights

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

# ---- Load voice & input ----
voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1]  # (1, 256)
style = torch.from_numpy(style_np).float()
print(f"Style: {style.shape}, range {style.min().item():.3f} to {style.max().item():.3f}")

# Simple token sequence (same as before)
input_ids = torch.tensor([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=torch.long)

# ---- Build PT model ----
print("Building PyTorch model...")
model = build_model(ONNX_PATH)
model.eval()

# ---- Run ONNX to get reference outputs ----
print("\nRunning ONNX inference...")
sess = ort.InferenceSession(ONNX_PATH)
onnx_inputs = {
    "input_ids": input_ids.numpy().astype(np.int64),
    "style": style_np.astype(np.float32),
    "speed": np.array([1.0], dtype=np.float32),
}
onnx_out = sess.run(None, onnx_inputs)
onnx_audio = onnx_out[0]   # (N_samples,) - waveform
onnx_dur = onnx_out[1]     # (T,) - durations
print(f"ONNX audio: shape {onnx_audio.shape}, range {onnx_audio.min():.4f} to {onnx_audio.max():.4f}")
print(f"ONNX durations: {onnx_dur}, sum={onnx_dur.sum()}")

# ---- Run PT step-by-step ----
with torch.no_grad():
    B, T = input_ids.shape

    # Text encoding
    text_proj, text_lstm = model.text_encoder(input_ids)
    print(f"\ntext_proj: {text_proj.shape}, range {text_proj.min():.4f} to {text_proj.max():.4f}")

    # ALBERT
    bert_out = model.bert(input_ids)
    print(f"bert_out: {bert_out.shape}, range {bert_out.min():.4f} to {bert_out.max():.4f}")

    # Predictor phoneme level
    text_enc_out, durations = model.predictor.forward(bert_out, style, speed=1.0)
    print(f"\ntext_enc_out: {text_enc_out.shape}, range {text_enc_out.min():.4f} to {text_enc_out.max():.4f}")
    print(f"durations: {durations[0]}, sum={durations[0].sum().item()}")

    # Length regulation
    dur_int = durations.round().long().clamp(min=1)
    lr_features = model._length_regulate(text_enc_out, dur_int)
    T_frames = lr_features.shape[1]
    print(f"\nLR features: {lr_features.shape}, range {lr_features.min():.4f} to {lr_features.max():.4f}")

    # Predictor frame level
    shared_h, f0_pred, n_pred = model.predictor.forward_frame(lr_features, style)
    print(f"\nshared_h: {shared_h.shape}, range {shared_h.min():.4f} to {shared_h.max():.4f}")
    print(f"f0_pred: {f0_pred.shape}, range {f0_pred.min():.4f} to {f0_pred.max():.4f}")
    print(f"n_pred: {n_pred.shape}, range {n_pred.min():.4f} to {n_pred.max():.4f}")

    # Acoustic decoder
    text_proj_lr = model._length_regulate_proj(text_proj, dur_int)
    features, f0_2T = model.decoder(text_proj_lr, f0_pred, n_pred, style)
    T2 = features.shape[-1]
    print(f"\nDecoder features: {features.shape}, range {features.min():.4f} to {features.max():.4f}")
    print(f"f0_2T: {f0_2T.shape}, range {f0_2T.min():.4f} to {f0_2T.max():.4f}")

    # Upsample F0
    N_approx = T2 * 300
    f0_samples = F.interpolate(f0_2T.float(), size=N_approx, mode='linear', align_corners=False)
    f0_hz = F.relu(f0_samples)

    # ---- Generator step-by-step ----
    gen = model.decoder.generator
    s = style[:, 128:]

    # SineGenerator
    sine = gen.sine_gen(f0_hz, deterministic=True)
    print(f"\nSine: {sine.shape}, range {sine.min():.4f} to {sine.max():.4f}")

    # Forward STFT of sine
    sine_stft = gen.stft.forward_stft(sine)
    print(f"sine_stft: {sine_stft.shape}, range {sine_stft.min():.4f} to {sine_stft.max():.4f}")

    noise_0 = gen.noise_convs[0](sine_stft)
    noise_1 = gen.noise_convs[1](sine_stft)
    n0 = gen.noise_res[0](noise_0, s)
    n1 = gen.noise_res[1](noise_1, s)
    print(f"noise_0: {noise_0.shape}, range {noise_0.min():.4f} to {noise_0.max():.4f}")
    print(f"n0 (after resblock): {n0.shape}, range {n0.min():.4f} to {n0.max():.4f}")

    # Stage 1
    x = F.leaky_relu(features, 0.2)
    x = gen.ups[0](x)
    from kitten_torch.modules.generator import _match_length, _match_length_pad
    n0_aligned = _match_length(n0, x.shape[-1])
    x = x + n0_aligned
    r0 = gen.resblocks[0](x, s)
    r1 = gen.resblocks[1](x, s)
    x = r0 + r1
    print(f"\nStage 1 output: {x.shape}, range {x.min():.4f} to {x.max():.4f}")

    # Stage 2
    x = F.leaky_relu(x, 0.2)
    x = gen.ups[1](x)
    n1_aligned = _match_length_pad(x, n1.shape[-1])
    x = n1_aligned + n1
    r2 = gen.resblocks[2](x, s)
    r3 = gen.resblocks[3](x, s)
    x = r2 + r3
    print(f"Stage 2 output: {x.shape}, range {x.min():.4f} to {x.max():.4f}")

    # Conv post
    x = F.leaky_relu(x, 0.2)
    x = gen.conv_post(x)
    print(f"conv_post output: {x.shape}, range {x.min():.4f} to {x.max():.4f}")

    # iSTFT
    audio = gen.stft.inverse_stft(x)
    print(f"iSTFT output: {audio.shape}, range {audio.min():.4f} to {audio.max():.4f}")

print(f"\n=== Summary ===")
print(f"ONNX audio: shape {onnx_audio.shape}, range {onnx_audio.min():.4f} to {onnx_audio.max():.4f}")
print(f"PT audio: range {audio.min():.4f} to {audio.max():.4f}")
print(f"Amplitude ratio (PT max / ONNX max): {audio.abs().max().item() / (abs(onnx_audio).max() + 1e-8):.3f}")
