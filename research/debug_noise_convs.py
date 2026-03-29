"""Feed ONNX stft directly to PT noise_convs to check weight loading."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from kitten_torch.model import build_model

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
    "/decoder/generator/Concat_1_output_0",            # forward STFT output [mag, phase]
    "/decoder/generator/noise_convs.0/Conv_output_0",  # noise_convs.0 output
    "/decoder/generator/noise_convs.1/Conv_output_0",  # noise_convs.1 output
    "/decoder/generator/noise_res.0/Add_8_output_0",   # noise_res.0 final
    "/decoder/generator/noise_res.1/Add_8_output_0",   # noise_res.1 final
]

model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)
for name in probe_names:
    try:
        elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
        vi = onnx.helper.make_tensor_value_info(name, elem_type, None)
        model_copy.graph.output.append(vi)
    except: pass

sess = ort.InferenceSession(model_copy.SerializeToString(), providers=["CPUExecutionProvider"])
results = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed})
output_names = [o.name for o in sess.get_outputs()]
onnx_vals = {}
for name, result in zip(output_names, results):
    if name in set(probe_names):
        arr = np.asarray(result, dtype=np.float32)
        onnx_vals[name] = arr

model = build_model(ONNX_PATH)
model.eval()
gen = model.decoder.generator
style = torch.from_numpy(style_np)
s = style[:, 128:]

# Feed ONNX stft directly to PT noise_convs
onnx_stft = onnx_vals["/decoder/generator/Concat_1_output_0"]
print(f"ONNX stft: shape={onnx_stft.shape}, [mag]: 0 to {onnx_stft[:,0:11,:].max():.4f}, [phase]: {onnx_stft[:,11:22,:].min():.4f} to {onnx_stft[:,11:22,:].max():.4f}")

stft_t = torch.from_numpy(onnx_stft)

with torch.no_grad():
    pt_nc0 = gen.noise_convs[0](stft_t)
    pt_nc1 = gen.noise_convs[1](stft_t)

onnx_nc0 = onnx_vals["/decoder/generator/noise_convs.0/Conv_output_0"]
onnx_nc1 = onnx_vals["/decoder/generator/noise_convs.1/Conv_output_0"]

print(f"\nWith ONNX stft input:")
print(f"  PT noise_convs.0: {pt_nc0.shape}, range={pt_nc0.min():.4f} to {pt_nc0.max():.4f}")
print(f"  ONNX noise_convs.0: {onnx_nc0.shape}, range={onnx_nc0.min():.4f} to {onnx_nc0.max():.4f}")
print(f"  MAE: {(pt_nc0.numpy() - onnx_nc0).abs().mean():.4f}")
print(f"  Corr: {np.corrcoef(pt_nc0.numpy().flatten(), onnx_nc0.flatten())[0,1]:.4f}")

print(f"\n  PT noise_convs.1: {pt_nc1.shape}, range={pt_nc1.min():.4f} to {pt_nc1.max():.4f}")
print(f"  ONNX noise_convs.1: {onnx_nc1.shape}, range={onnx_nc1.min():.4f} to {onnx_nc1.max():.4f}")
print(f"  MAE: {(pt_nc1.numpy() - onnx_nc1).abs().mean():.4f}")
print(f"  Corr: {np.corrcoef(pt_nc1.numpy().flatten(), onnx_nc1.flatten())[0,1]:.4f}")

# noise_res with ONNX nc inputs
with torch.no_grad():
    pt_nr0 = gen.noise_res[0](pt_nc0, s)
    pt_nr1 = gen.noise_res[1](pt_nc1, s)

onnx_nr0 = onnx_vals["/decoder/generator/noise_res.0/Add_8_output_0"]
onnx_nr1 = onnx_vals["/decoder/generator/noise_res.1/Add_8_output_0"]

print(f"\nWith ONNX stft → noise_res:")
print(f"  PT noise_res.0: {pt_nr0.shape}, range={pt_nr0.min():.4f} to {pt_nr0.max():.4f}")
print(f"  ONNX noise_res.0: {onnx_nr0.shape}, range={onnx_nr0.min():.4f} to {onnx_nr0.max():.4f}")
print(f"  MAE: {(pt_nr0.numpy() - onnx_nr0).abs().mean():.4f}")
print(f"  Corr: {np.corrcoef(pt_nr0.numpy().flatten(), onnx_nr0.flatten())[0,1]:.4f}")

print(f"\n  PT noise_res.1: {pt_nr1.shape}, range={pt_nr1.min():.4f} to {pt_nr1.max():.4f}")
print(f"  ONNX noise_res.1: {onnx_nr1.shape}, range={onnx_nr1.min():.4f} to {onnx_nr1.max():.4f}")
print(f"  MAE: {(pt_nr1.numpy() - onnx_nr1).abs().mean():.4f}")
print(f"  Corr: {np.corrcoef(pt_nr1.numpy().flatten(), onnx_nr1.flatten())[0,1]:.4f}")

# Check noise_convs weights for quantization
print("\n=== noise_convs weight quantization in ONNX ===")
init_dict = {init.name: np.array(onnx.numpy_helper.to_array(init)) for init in model_proto.graph.initializer}
for name, arr in init_dict.items():
    if 'noise_convs' in name:
        print(f"  {name!r}: shape={arr.shape}, dtype={arr.dtype}, range={arr.min():.4f} to {arr.max():.4f}")
