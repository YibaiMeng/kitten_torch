"""Verify forward_stft fix and compare noise_convs with ONNX using matched ONNX features."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from kitten_torch.model import build_model
from kitten_torch.modules.generator import _match_length, _match_length_pad

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids_np = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

model_proto = onnx.load(ONNX_PATH)
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}

# We'll feed the ONNX F0 (f0_upsamp output) directly to PT sine generator
# and compare the stft outputs and noise_convs outputs

probe_names = [
    "/decoder/generator/f0_upsamp/Resize_output_0",   # upsampled F0 (in Hz)
    "/decoder/generator/Concat_1_output_0",             # forward STFT output [mag, phase]
    "/decoder/generator/noise_convs.0/Conv_output_0",  # noise_convs.0 output
    "/decoder/generator/noise_convs.1/Conv_output_0",  # noise_convs.1 output
    "/decoder/generator/noise_res.0/Add_8_output_0",   # noise_res.0 final output
    "/decoder/generator/noise_res.1/Add_8_output_0",   # noise_res.1 final output
    "/decoder/generator/ups.0/ConvTranspose_output_0_Cast_to_float32_output_0",
    "/decoder/generator/Add_3_output_0",  # ups.0 + noise_res.0 = stage1 input
    "/decoder/generator/Div_1_output_0",  # MRF stage1 avg
    "/decoder/generator/Add_5_output_0",  # ups.1 + noise_res.1 = stage2 input
    "/decoder/generator/Div_2_output_0",  # MRF stage2 avg
    "/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_output_0",  # conv_post
    "/decoder/decode.3/Mul_output_0",                  # decoder features
]

model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)
for name in probe_names:
    try:
        elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
        vi = onnx.helper.make_tensor_value_info(name, elem_type, None)
        model_copy.graph.output.append(vi)
    except:
        pass

sess = ort.InferenceSession(model_copy.SerializeToString(), providers=["CPUExecutionProvider"])
results = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed})
output_names = [o.name for o in sess.get_outputs()]
onnx_vals = {}
print("=== ONNX intermediates ===")
for name, result in zip(output_names, results):
    if name in set(probe_names):
        arr = np.asarray(result, dtype=np.float32)
        onnx_vals[name] = arr
        print(f"  {name.split('/')[-1]}: shape={arr.shape}, range={arr.min():.4f} to {arr.max():.4f}")

# Also get final ONNX audio
sess2 = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_audio = sess2.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed})[0]

# --- PT model ---
model = build_model(ONNX_PATH)
model.eval()
gen = model.decoder.generator
style = torch.from_numpy(style_np)
s = style[:, 128:]

print("\n=== PT using ONNX F0 and ONNX features ===")
# Get ONNX F0 (already upsampled, in Hz form)
onnx_f0_up = onnx_vals.get("/decoder/generator/f0_upsamp/Resize_output_0")
if onnx_f0_up is not None:
    print(f"ONNX f0_upsamp: shape={onnx_f0_up.shape}, range={onnx_f0_up.min():.4f} to {onnx_f0_up.max():.4f}")
    # The f0_upsamp in ONNX is (1, 1, N_samples) or (B, N)?
    # Feed this to sine_gen
    f0_t = torch.from_numpy(onnx_f0_up)
    if f0_t.dim() == 2:
        f0_t = f0_t.unsqueeze(1)  # (B, 1, N)
    print(f"f0 for sine gen: {f0_t.shape}")

    with torch.no_grad():
        sine = gen.sine_gen(f0_t, deterministic=True)
        print(f"PT sine: {sine.shape}, range={sine.min():.4f} to {sine.max():.4f}")

        sine_stft = gen.stft.forward_stft(sine)
        print(f"PT sine_stft [mag,phase]: {sine_stft.shape}, range={sine_stft.min():.4f} to {sine_stft.max():.4f}")
        print(f"  channels [0:11] (magnitude): {sine_stft[:,0:11,:].min():.4f} to {sine_stft[:,0:11,:].max():.4f}")
        print(f"  channels [11:22] (phase): {sine_stft[:,11:22,:].min():.4f} to {sine_stft[:,11:22,:].max():.4f}")

        onnx_stft = onnx_vals.get("/decoder/generator/Concat_1_output_0")
        if onnx_stft is not None:
            print(f"ONNX stft [mag,phase]: shape={onnx_stft.shape}")
            stft_t = torch.from_numpy(onnx_stft)
            # Size may differ due to different padding; align
            min_T = min(sine_stft.shape[-1], stft_t.shape[-1])
            mae = (sine_stft[:,:,:min_T] - stft_t[:,:,:min_T]).abs().mean().item()
            print(f"  MAE vs ONNX (first {min_T} frames): {mae:.4f}")

        # noise_convs with our PT STFT output
        nc0 = gen.noise_convs[0](sine_stft)
        nc1 = gen.noise_convs[1](sine_stft)
        print(f"\nPT noise_convs.0: {nc0.shape}, range={nc0.min():.4f} to {nc0.max():.4f}")
        print(f"PT noise_convs.1: {nc1.shape}, range={nc1.min():.4f} to {nc1.max():.4f}")
        onnx_nc0 = onnx_vals.get("/decoder/generator/noise_convs.0/Conv_output_0")
        onnx_nc1 = onnx_vals.get("/decoder/generator/noise_convs.1/Conv_output_0")
        if onnx_nc0 is not None:
            print(f"ONNX noise_convs.0: shape={onnx_nc0.shape}, range={onnx_nc0.min():.4f} to {onnx_nc0.max():.4f}")
            min_T = min(nc0.shape[-1], onnx_nc0.shape[-1])
            mae = (nc0[:,:,:min_T] - torch.from_numpy(onnx_nc0[:,:,:min_T])).abs().mean().item()
            print(f"  MAE vs ONNX (first {min_T}): {mae:.4f}")
        if onnx_nc1 is not None:
            print(f"ONNX noise_convs.1: shape={onnx_nc1.shape}, range={onnx_nc1.min():.4f} to {onnx_nc1.max():.4f}")
            min_T = min(nc1.shape[-1], onnx_nc1.shape[-1])
            mae = (nc1[:,:,:min_T] - torch.from_numpy(onnx_nc1[:,:,:min_T])).abs().mean().item()
            print(f"  MAE vs ONNX (first {min_T}): {mae:.4f}")

        # noise_res
        n0 = gen.noise_res[0](nc0, s)
        n1 = gen.noise_res[1](nc1, s)
        print(f"\nPT noise_res.0: {n0.shape}, range={n0.min():.4f} to {n0.max():.4f}")
        print(f"PT noise_res.1: {n1.shape}, range={n1.min():.4f} to {n1.max():.4f}")
        onnx_nr0 = onnx_vals.get("/decoder/generator/noise_res.0/Add_8_output_0")
        onnx_nr1 = onnx_vals.get("/decoder/generator/noise_res.1/Add_8_output_0")
        if onnx_nr0 is not None:
            min_T = min(n0.shape[-1], onnx_nr0.shape[-1])
            mae = (n0[:,:,:min_T] - torch.from_numpy(onnx_nr0[:,:,:min_T])).abs().mean().item()
            print(f"  noise_res.0 MAE: {mae:.4f}, ONNX range={onnx_nr0.min():.4f} to {onnx_nr0.max():.4f}")
        if onnx_nr1 is not None:
            min_T = min(n1.shape[-1], onnx_nr1.shape[-1])
            mae = (n1[:,:,:min_T] - torch.from_numpy(onnx_nr1[:,:,:min_T])).abs().mean().item()
            print(f"  noise_res.1 MAE: {mae:.4f}, ONNX range={onnx_nr1.min():.4f} to {onnx_nr1.max():.4f}")

# Now run the full PT model and compare audio
print("\n=== Full PT inference ===")
input_ids = torch.from_numpy(input_ids_np)
with torch.no_grad():
    audio_pt = model(input_ids, style, speed=1.0)

print(f"PT audio: shape={audio_pt.shape}, range={audio_pt.min():.4f} to {audio_pt.max():.4f}")
print(f"ONNX audio: shape={onnx_audio.shape}, range={onnx_audio.min():.4f} to {onnx_audio.max():.4f}")
print(f"PT std={audio_pt.std():.4f}, ONNX std={onnx_audio.std():.4f}")
print(f"Amplitude ratio (PT/ONNX std): {audio_pt.std().item() / (onnx_audio.std() + 1e-8):.3f}")
