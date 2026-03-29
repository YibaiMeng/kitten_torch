"""
Feed ONNX intermediates directly to PT main signal path to isolate amplitude issue.
Use ONNX: features, ups.0, noise_res.0, MRF stage1, ups.1, noise_res.1, etc.
"""
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

probe_names = [
    "/decoder/decode.3/Mul_output_0",                  # decoder features (256, 54)
    "/decoder/generator/ups.0/ConvTranspose_output_0_Cast_to_float32_output_0",  # ups.0 output
    "/decoder/generator/noise_res.0/Add_8_output_0",   # noise_res.0 output
    "/decoder/generator/Add_3_output_0",               # ups.0 + noise_res.0
    "/decoder/generator/resblocks.0/Add_8_output_0",   # resblocks.0 output
    "/decoder/generator/resblocks.1/Add_8_output_0",   # resblocks.1 output
    "/decoder/generator/Div_1_output_0",               # MRF stage1 avg
    "/decoder/generator/ups.1/ConvTranspose_output_0_Cast_to_float32_output_0",  # ups.1 output
    "/decoder/generator/noise_res.1/Add_8_output_0",   # noise_res.1 output
    "/decoder/generator/Add_5_output_0",               # ups.1 + noise_res.1 (stage2 input)
    "/decoder/generator/resblocks.2/Add_8_output_0",   # resblocks.2 output
    "/decoder/generator/resblocks.3/Add_8_output_0",   # resblocks.3 output
    "/decoder/generator/Div_2_output_0",               # MRF stage2 avg
    "/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_output_0",  # conv_post
]

model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)
added = []
for name in probe_names:
    try:
        elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
        vi = onnx.helper.make_tensor_value_info(name, elem_type, None)
        model_copy.graph.output.append(vi)
        added.append(name)
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

# Get ONNX audio
sess2 = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_audio = sess2.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed})[0]

model = build_model(ONNX_PATH)
model.eval()
gen = model.decoder.generator
style = torch.from_numpy(style_np)
s = style[:, 128:]

print("\n=== PT with ONNX intermediates directly ===")

def cmp(pt_t, onnx_arr, name):
    diff = np.abs(pt_t.numpy() - onnx_arr)
    print(f"  {name}: MAE={diff.mean():.4f}, max={diff.max():.4f}")
    print(f"    PT: {pt_t.min():.4f} to {pt_t.max():.4f}")
    print(f"    ONNX: {onnx_arr.min():.4f} to {onnx_arr.max():.4f}")

with torch.no_grad():
    # Stage 1: features → LeakyReLU → ups.0
    features = torch.from_numpy(onnx_vals["/decoder/decode.3/Mul_output_0"])
    x_lr = F.leaky_relu(features, 0.1)
    ups0_pt = gen.ups[0](x_lr)
    ups0_onnx = onnx_vals["/decoder/generator/ups.0/ConvTranspose_output_0_Cast_to_float32_output_0"]
    cmp(ups0_pt, ups0_onnx, "ups.0")

    # Stage 1: ups.0 + ONNX noise_res.0 (bypass PT noise path)
    nr0_onnx = torch.from_numpy(onnx_vals["/decoder/generator/noise_res.0/Add_8_output_0"])
    ups0_t = torch.from_numpy(ups0_onnx)
    stage1_input = ups0_t + nr0_onnx
    stage1_input_onnx = onnx_vals["/decoder/generator/Add_3_output_0"]
    cmp(stage1_input, stage1_input_onnx, "stage1 input (ups0+nr0)")

    # Resblocks.0 and .1 with ONNX stage1 input
    stage1_input_t = torch.from_numpy(stage1_input_onnx)
    r0 = gen.resblocks[0](stage1_input_t, s)
    r1 = gen.resblocks[1](stage1_input_t, s)
    r0_onnx = onnx_vals.get("/decoder/generator/resblocks.0/Add_8_output_0")
    r1_onnx = onnx_vals.get("/decoder/generator/resblocks.1/Add_8_output_0")
    if r0_onnx is not None:
        cmp(r0, r0_onnx, "resblocks.0")
    if r1_onnx is not None:
        cmp(r1, r1_onnx, "resblocks.1")

    # MRF avg
    mrf1 = (r0 + r1) * 0.5
    mrf1_onnx = onnx_vals.get("/decoder/generator/Div_1_output_0")
    if mrf1_onnx is not None:
        cmp(mrf1, mrf1_onnx, "MRF stage1")

    # Stage 2
    mrf1_t = torch.from_numpy(mrf1_onnx) if mrf1_onnx is not None else mrf1
    x2_lr = F.leaky_relu(mrf1_t, 0.1)
    ups1_pt = gen.ups[1](x2_lr)
    ups1_onnx = onnx_vals.get("/decoder/generator/ups.1/ConvTranspose_output_0_Cast_to_float32_output_0")
    if ups1_onnx is not None:
        cmp(ups1_pt, ups1_onnx, "ups.1")

    # ups.1 + ONNX noise_res.1
    nr1_onnx = onnx_vals["/decoder/generator/noise_res.1/Add_8_output_0"]
    nr1_t = torch.from_numpy(nr1_onnx)
    ups1_t = torch.from_numpy(ups1_onnx) if ups1_onnx is not None else ups1_pt
    # Reflection pad
    x2_padded = _match_length_pad(ups1_t, nr1_t.shape[-1])
    stage2_input = x2_padded + nr1_t
    stage2_input_onnx = onnx_vals.get("/decoder/generator/Add_5_output_0")
    if stage2_input_onnx is not None:
        cmp(stage2_input, stage2_input_onnx, "stage2 input (ups1+nr1)")

    stage2_in_t = torch.from_numpy(stage2_input_onnx) if stage2_input_onnx is not None else stage2_input
    r2 = gen.resblocks[2](stage2_in_t, s)
    r3 = gen.resblocks[3](stage2_in_t, s)
    r2_onnx = onnx_vals.get("/decoder/generator/resblocks.2/Add_8_output_0")
    r3_onnx = onnx_vals.get("/decoder/generator/resblocks.3/Add_8_output_0")
    if r2_onnx is not None:
        cmp(r2, r2_onnx, "resblocks.2")
    if r3_onnx is not None:
        cmp(r3, r3_onnx, "resblocks.3")

    mrf2 = (r2 + r3) * 0.5
    mrf2_onnx = onnx_vals.get("/decoder/generator/Div_2_output_0")
    if mrf2_onnx is not None:
        cmp(mrf2, mrf2_onnx, "MRF stage2")

    # Conv post + iSTFT
    mrf2_t = torch.from_numpy(mrf2_onnx) if mrf2_onnx is not None else mrf2
    x3_lr = F.leaky_relu(mrf2_t, 0.01)
    cp_pt = gen.conv_post(x3_lr)
    cp_onnx = onnx_vals.get("/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_output_0")
    if cp_onnx is not None:
        cmp(cp_pt, cp_onnx, "conv_post")
        print(f"  conv_post log_mag: PT max={cp_pt[:,0:11,:].max():.4f}, ONNX max={cp_onnx[:,0:11,:].max():.4f}")
        print(f"  conv_post phase:   PT max={cp_pt[:,11:22,:].max():.4f}, ONNX max={cp_onnx[:,11:22,:].max():.4f}")

    # iSTFT
    cp_t = torch.from_numpy(cp_onnx) if cp_onnx is not None else cp_pt
    audio_pt = gen.stft.inverse_stft(cp_t)
    print(f"\n  iSTFT(ONNX conv_post): range={audio_pt.min():.4f} to {audio_pt.max():.4f}, std={audio_pt.std():.4f}")
    print(f"  ONNX audio: range={onnx_audio.min():.4f} to {onnx_audio.max():.4f}, std={onnx_audio.std():.4f}")
    print(f"  Amplitude ratio (PT/ONNX): {audio_pt.std().item() / onnx_audio.std():.4f}")

    # Also test iSTFT with PT conv_post
    audio_pt2 = gen.stft.inverse_stft(cp_pt)
    print(f"\n  iSTFT(PT conv_post): range={audio_pt2.min():.4f} to {audio_pt2.max():.4f}, std={audio_pt2.std():.4f}")
    print(f"  Amplitude ratio (PT/ONNX): {audio_pt2.std().item() / onnx_audio.std():.4f}")
