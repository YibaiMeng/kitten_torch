"""
Run the full PT pipeline and compare key intermediate values against ONNX.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort

from kitten_torch.model import build_model

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
style = torch.from_numpy(style_np)
input_ids = torch.tensor([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=torch.long)

# ONNX run
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_audio, onnx_dur = sess.run(None, {
    "input_ids": input_ids.numpy(),
    "style": style_np,
    "speed": np.array([1.0], dtype=np.float32),
})
print(f"ONNX: audio shape={onnx_audio.shape}, range={onnx_audio.min():.4f} to {onnx_audio.max():.4f}")
print(f"ONNX: durations sum={onnx_dur.sum()}, values={onnx_dur}")

# PT run with hooks to capture intermediates
print("\nBuilding PT model...")
model = build_model(ONNX_PATH)
model.eval()

# Hook to capture generator conv_post output
captured = {}

def hook_conv_post(module, input, output):
    captured['conv_post_out'] = output.detach().cpu()

model.decoder.generator.conv_post.register_forward_hook(hook_conv_post)

with torch.no_grad():
    # Run text encoder + BERT + predictor + decoder + generator
    text_proj, text_lstm = model.text_encoder(input_ids)
    bert_out = model.bert(input_ids)
    text_enc_out, durations = model.predictor.forward(bert_out, style, 1.0)
    dur_int = durations.round().long().clamp(min=1)

    print(f"\nPT durations: sum={dur_int.sum().item()}, values={dur_int[0].tolist()}")

    lr_features = model._length_regulate(text_enc_out, dur_int)
    shared_h, f0_pred, n_pred = model.predictor.forward_frame(lr_features, style)
    text_proj_lr = model._length_regulate_proj(text_proj, dur_int)
    features, f0_2T = model.decoder(text_proj_lr, f0_pred, n_pred, style)

    print(f"\nDecoder features: shape={features.shape}, range={features.min():.4f} to {features.max():.4f}")

    T2 = features.shape[-1]
    N_approx = T2 * 300
    f0_samples = F.interpolate(f0_2T.float(), size=N_approx, mode='linear', align_corners=False)
    f0_hz = F.relu(f0_samples)

    # Full generator
    audio = model.decoder.generator(features, f0_hz, style, deterministic=True)

print(f"\nPT audio: shape={audio.shape}, range={audio.min():.4f} to {audio.max():.4f}")

# Analyze conv_post output
if 'conv_post_out' in captured:
    cp = captured['conv_post_out']
    print(f"\nconv_post output: shape={cp.shape}, range={cp.min():.4f} to {cp.max():.4f}")
    log_mag = cp[:, :11, :]
    phase = cp[:, 11:, :]
    print(f"  [0:11] log_mag: range={log_mag.min():.4f} to {log_mag.max():.4f}, abs_mean={log_mag.abs().mean():.4f}")
    print(f"  [11:22] phase:  range={phase.min():.4f} to {phase.max():.4f}, abs_mean={phase.abs().mean():.4f}")

    # What would polar iSTFT give?
    mag = torch.exp(log_mag)
    print(f"  exp(log_mag): range={mag.min():.4f} to {mag.max():.4f}")

print(f"\nAmplitude ratio: PT_max/ONNX_max = {audio.abs().max().item() / abs(onnx_audio).max():.3f}")
