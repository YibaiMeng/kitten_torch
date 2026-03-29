"""Debug generator with ONNX features to isolate explosion."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from kitten_torch.model import build_model

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids_np = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

# Get ONNX features and intermediates
model_proto = onnx.load(ONNX_PATH)
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}
model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)

probe_names = [
    "/decoder/decode.3/Mul_output_0",
    "/decoder/generator/Add_3_output_0",  # ups.0 + noise_rb0 = input to stage2
    "/decoder/generator/ups.0/ConvTranspose_output_0",
    "/decoder/generator/resblocks.0/Add_2_output_0",  # resblocks.0 output after 3 dilations
    "/decoder/generator/resblocks.1/Add_2_output_0",  # resblocks.1 output
    "/decoder/generator/Div_1_output_0",  # MRF stage1 avg
]
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
        print(f"ONNX {name.split('/')[-1]}: shape={arr.shape}, range={arr.min():.4f} to {arr.max():.4f}")

# PT model with hooks at each stage
model = build_model(ONNX_PATH)
model.eval()
gen = model.decoder.generator

# Monkey-patch to add print debugging
orig_forward = gen.forward

def debug_forward(features, f0_samples, style, deterministic=False):
    s = style[:, 128:]
    sine = gen.sine_gen(f0_samples, deterministic)
    sine_stft = gen.stft.forward_stft(sine)
    noise_0 = gen.noise_convs[0](sine_stft)
    noise_1 = gen.noise_convs[1](sine_stft)
    n0 = gen.noise_res[0](noise_0, s)
    n1 = gen.noise_res[1](noise_1, s)
    print(f"\nPT gen stages (with ONNX features):")
    print(f"  sine_stft: {sine_stft.shape}, {sine_stft.min():.4f} to {sine_stft.max():.4f}")
    print(f"  noise_0: {noise_0.shape}, {noise_0.min():.4f} to {noise_0.max():.4f}")
    print(f"  n0 (noise_res.0): {n0.shape}, {n0.min():.4f} to {n0.max():.4f}")

    x = F.leaky_relu(features, 0.2)
    x = gen.ups[0](x)
    print(f"  ups.0 (before noise): {x.shape}, {x.min():.4f} to {x.max():.4f}")
    from kitten_torch.modules.generator import _match_length
    n0m = _match_length(n0, x.shape[-1])
    x = x + n0m
    print(f"  after noise_add: {x.min():.4f} to {x.max():.4f}")

    r0 = gen.resblocks[0](x, s)
    print(f"  resblocks.0: {r0.shape}, {r0.min():.4f} to {r0.max():.4f}")
    r1 = gen.resblocks[1](x, s)
    print(f"  resblocks.1: {r1.shape}, {r1.min():.4f} to {r1.max():.4f}")
    x = (r0 + r1) * 0.5
    print(f"  MRF stage1: {x.min():.4f} to {x.max():.4f}")

    x = F.leaky_relu(x, 0.2)
    x = gen.ups[1](x)
    print(f"  ups.1: {x.shape}, {x.min():.4f} to {x.max():.4f}")
    from kitten_torch.modules.generator import _match_length_pad
    x = _match_length_pad(x, n1.shape[-1])
    x = x + n1
    print(f"  after noise_add2: {x.min():.4f} to {x.max():.4f}")

    r2 = gen.resblocks[2](x, s)
    print(f"  resblocks.2: {r2.shape}, {r2.min():.4f} to {r2.max():.4f}")
    r3 = gen.resblocks[3](x, s)
    print(f"  resblocks.3: {r3.shape}, {r3.min():.4f} to {r3.max():.4f}")
    x = (r2 + r3) * 0.5
    print(f"  MRF stage2: {x.min():.4f} to {x.max():.4f}")

    x = F.leaky_relu(x, 0.2)
    x = gen.conv_post(x)
    print(f"  conv_post: {x.min():.4f} to {x.max():.4f}")
    log_mag = x[:, :11, :]
    print(f"  log_mag max: {log_mag.max():.4f}, exp(max)={log_mag.max().exp():.4f}")
    audio = gen.stft.inverse_stft(x)
    return audio

# Feed ONNX decode.3 output to PT generator with zero F0
onnx_features = onnx_vals.get("/decoder/decode.3/Mul_output_0")
if onnx_features is not None:
    features = torch.from_numpy(onnx_features)
    N = onnx_features.shape[-1] * 300
    f0_zero = torch.zeros(1, 1, N)
    style = torch.from_numpy(style_np)
    with torch.no_grad():
        audio = debug_forward(features, f0_zero, style, deterministic=True)
    print(f"\nFinal audio: {audio.min():.4f} to {audio.max():.4f}")
