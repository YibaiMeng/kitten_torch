"""
Check what ONNX conv_post channels [0:11] vs [11:22] actually contain,
and verify if polar iSTFT interpretation is correct.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

model_proto = onnx.load(ONNX_PATH)
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}

# Probe conv_post output, Slice_1/2, Exp output, final Sub_1 (audio before trim)
probe_names = [
    "/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_output_0",
    "/decoder/generator/Slice_1_output_0",  # [0:11] or [11:22]
    "/decoder/generator/Slice_2_output_0",  # the other slice
    "/decoder/generator/Exp_output_0",
    "/decoder/generator/Sin_output_0",
    "/decoder/generator/Sub_1_output_0",
]

model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)
for name in probe_names:
    elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
    vi = onnx.helper.make_tensor_value_info(name, elem_type, None)
    model_copy.graph.output.append(vi)

sess = ort.InferenceSession(model_copy.SerializeToString(), providers=["CPUExecutionProvider"])
results = sess.run(None, {"input_ids": input_ids, "style": style_np, "speed": speed})
output_names = [o.name for o in sess.get_outputs()]
probe_set = set(probe_names)
onnx_values = {}
for name, result in zip(output_names, results):
    if name in probe_set:
        arr = np.asarray(result, dtype=np.float32)
        onnx_values[name] = arr
        print(f"{name!r}: shape={arr.shape}")
        print(f"  range: {arr.min():.6f} to {arr.max():.6f}")
        print(f"  abs mean: {np.abs(arr).mean():.4f}")

# Also probe what the actual audio looks like from Sub_1
print(f"\nSub_1 (audio before trim): shape={onnx_values.get('/decoder/generator/Sub_1_output_0', np.array([])).shape}")

# Check the ONNX iSTFT manually using our loaded weights
print("\n=== Manual ONNX iSTFT reconstruction ===")
from kitten_torch.model import build_model
model = build_model(ONNX_PATH)
model.eval()

stft = model.decoder.generator.stft

conv_post_out = onnx_values["/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_output_0"]
slice1 = onnx_values["/decoder/generator/Slice_1_output_0"]  # goes to Exp
slice2 = onnx_values["/decoder/generator/Slice_2_output_0"]  # goes to Sin
exp_out = onnx_values["/decoder/generator/Exp_output_0"]
sin_out = onnx_values["/decoder/generator/Sin_output_0"]

print(f"\nSlice_1 (→Exp): range {slice1.min():.4f} to {slice1.max():.4f}")
print(f"Slice_2 (→Sin): range {slice2.min():.4f} to {slice2.max():.4f}")
print(f"Exp output: range {exp_out.min():.4f} to {exp_out.max():.4f}")
print(f"Sin output: range {sin_out.min():.4f} to {sin_out.max():.4f}")

# Which channels are Slice_1 (→Exp) and which are Slice_2 (→Sin)?
print(f"\nconv_post channels [0:11] range: {conv_post_out[:,0:11,:].min():.4f} to {conv_post_out[:,0:11,:].max():.4f}")
print(f"conv_post channels [11:22] range: {conv_post_out[:,11:22,:].min():.4f} to {conv_post_out[:,11:22,:].max():.4f}")

# Check if slice1 = channels [0:11] or [11:22]
if np.allclose(slice1, conv_post_out[:,0:11,:], atol=1e-4):
    print("Slice_1 = channels [0:11] (log-magnitude)")
    print("Slice_2 = channels [11:22] (phase)")
elif np.allclose(slice1, conv_post_out[:,11:22,:], atol=1e-4):
    print("Slice_1 = channels [11:22]")
    print("Slice_2 = channels [0:11]")
else:
    print("Slice_1 doesn't match [0:11] or [11:22] cleanly")
    print(f"  vs [0:11] max diff: {np.abs(slice1 - conv_post_out[:,0:11,:]).max():.4f}")
    print(f"  vs [11:22] max diff: {np.abs(slice1 - conv_post_out[:,11:22,:]).max():.4f}")

# Now compute ONNX iSTFT manually with our PT weights
# But use correct ONNX procedure (polar)
spec = torch.from_numpy(conv_post_out)
with torch.no_grad():
    # Approach 1: Direct real/imag
    audio_direct = stft.inverse_stft(spec)

    # Approach 2: Polar form as per ONNX
    log_mag = torch.from_numpy(slice1)   # channels going to Exp
    phase = torch.from_numpy(slice2)     # channels going to Sin
    mag = torch.exp(log_mag)
    phase_sin = torch.sin(phase)
    real = mag * torch.cos(phase_sin)
    imag = mag * torch.sin(phase_sin)
    real_out = F.conv_transpose1d(real, stft.weight_backward_real, stride=stft.hop)
    imag_out = F.conv_transpose1d(imag, stft.weight_backward_imag, stride=stft.hop)
    audio_polar = real_out - imag_out

print(f"\nDirect iSTFT: range {audio_direct.min():.4f} to {audio_direct.max():.4f}")
print(f"Polar iSTFT (with ONNX spec): range {audio_polar.min():.4f} to {audio_polar.max():.4f}")

# Get ONNX actual audio
sess2 = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_audio = sess2.run(None, {"input_ids": input_ids, "style": style_np, "speed": speed})[0]
print(f"ONNX actual audio: range {onnx_audio.min():.4f} to {onnx_audio.max():.4f}")
