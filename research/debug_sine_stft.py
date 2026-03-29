"""
Probe ONNX sine generator output (with Tanh) and forward STFT output
to understand what PT is computing differently.
"""
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
    # Sine generator output (after Tanh)
    "/decoder/generator/m_source/l_tanh/Tanh_output_0",
    # sine generator linear output (before Tanh)
    "/decoder/generator/m_source/l_linear/Add_output_0",
    # Forward STFT raw conv outputs
    "/decoder/generator/Conv_output_0",
    "/decoder/generator/Conv_1_output_0",
    # sqrt(real^2 + imag^2) for magnitude
    "/decoder/generator/Sqrt_output_0",
    # atan2 for phase
    "/decoder/generator/Atan_output_0",
    # Concat_1: forward STFT output (mag+phase)
    "/decoder/generator/Concat_1_output_0",
    # noise_convs outputs
    "/decoder/generator/noise_convs.0/Conv_output_0",
    "/decoder/generator/noise_convs.1/Conv_output_0",
    # noise_res outputs
    "/decoder/generator/noise_res.0/Add_2_output_0",  # final Add in noise_res.0
    "/decoder/generator/noise_res.1/Add_2_output_0",  # final Add in noise_res.1
]

model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)
added = []
for name in probe_names:
    elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
    vi = onnx.helper.make_tensor_value_info(name, elem_type, None)
    model_copy.graph.output.append(vi)
    added.append(name)

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

# Also check add_2 nodes in noise_res
print("\n=== Looking for noise_res Add_2 nodes ===")
for node in model_proto.graph.node:
    if '/decoder/generator/noise_res' in node.name and node.op_type == 'Add':
        if node.name.endswith('Add_2') or 'Add_2' in node.name:
            print(f"  {node.name!r}: out={list(node.output)}")

# Check what noise_res.0's final node is
print("\n=== noise_res.0 last Conv output ===")
for node in model_proto.graph.node:
    if '/decoder/generator/noise_res.0/' in node.name and node.op_type in ['Add', 'Conv']:
        if 'conv' in node.name.lower() or 'Add_' in node.name:
            pass
    if node.name == '/decoder/generator/noise_res.0/convs2.2/Conv':
        print(f"  noise_res.0/convs2.2/Conv out={list(node.output)}")

# Find residual add outputs
print("\n=== noise_res residual adds ===")
for node in model_proto.graph.node:
    name = node.name
    if node.op_type == 'Add' and '/noise_res' in name:
        # Only top-level adds (residual connection adds)
        depth = name.count('/')
        if depth <= 5:
            print(f"  {name!r}: out={list(node.output)}")

# PT model comparison
print("\n=== PT comparison ===")
model = build_model(ONNX_PATH)
model.eval()
gen = model.decoder.generator

input_ids = torch.from_numpy(input_ids_np)
style = torch.from_numpy(style_np)

with torch.no_grad():
    text_proj, _ = model.text_encoder(input_ids)
    bert_out = model.bert(input_ids)
    text_enc_out, durations = model.predictor.forward(bert_out, style, 1.0)
    dur_int = durations.round().long().clamp(min=1)
    lr_features = model._length_regulate(text_enc_out, dur_int)
    shared_h, f0_pred, n_pred = model.predictor.forward_frame(lr_features, style)
    text_proj_lr = model._length_regulate_proj(text_proj, dur_int)
    features, f0_2T = model.decoder(text_proj_lr, f0_pred, n_pred, style)
    T2 = features.shape[-1]
    N_approx = T2 * 300
    f0_samples = F.interpolate(f0_2T.float(), size=N_approx, mode='linear', align_corners=False)
    f0_hz = F.relu(f0_samples)

    s = style[:, 128:]

    # PT sine generator (no tanh)
    sine = gen.sine_gen(f0_hz, deterministic=True)
    print(f"PT sine (no tanh): {sine.shape}, range={sine.min():.4f} to {sine.max():.4f}")

    # What would tanh do?
    sine_tanh = torch.tanh(sine)
    print(f"PT sine (with tanh): range={sine_tanh.min():.4f} to {sine_tanh.max():.4f}")

    # ONNX sine values
    onnx_tanh = onnx_vals.get("/decoder/generator/m_source/l_tanh/Tanh_output_0")
    if onnx_tanh is not None:
        print(f"ONNX sine (after tanh): shape={onnx_tanh.shape}, range={onnx_tanh.min():.4f} to {onnx_tanh.max():.4f}")

    # PT forward STFT
    sine_stft_pt = gen.stft.forward_stft(sine)
    print(f"\nPT sine_stft (from sine without tanh): {sine_stft_pt.shape}, range={sine_stft_pt.min():.4f} to {sine_stft_pt.max():.4f}")

    # ONNX forward STFT (Concat_1)
    onnx_stft = onnx_vals.get("/decoder/generator/Concat_1_output_0")
    if onnx_stft is not None:
        print(f"ONNX stft (mag+phase): shape={onnx_stft.shape}, range={onnx_stft.min():.4f} to {onnx_stft.max():.4f}")

    # Check what kind of STFT ONNX uses (magnitude/phase vs real/imag)
    onnx_sqrt = onnx_vals.get("/decoder/generator/Sqrt_output_0")
    onnx_atan = onnx_vals.get("/decoder/generator/Atan_output_0")
    if onnx_sqrt is not None:
        print(f"\nONNX STFT magnitude (Sqrt): shape={onnx_sqrt.shape}, range={onnx_sqrt.min():.4f} to {onnx_sqrt.max():.4f}")
    if onnx_atan is not None:
        print(f"ONNX STFT phase (Atan): shape={onnx_atan.shape}, range={onnx_atan.min():.4f} to {onnx_atan.max():.4f}")

    # noise_convs
    nc0 = gen.noise_convs[0](sine_stft_pt)
    nc1 = gen.noise_convs[1](sine_stft_pt)
    print(f"\nPT noise_convs.0: {nc0.shape}, range={nc0.min():.4f} to {nc0.max():.4f}")
    print(f"PT noise_convs.1: {nc1.shape}, range={nc1.min():.4f} to {nc1.max():.4f}")

    onnx_nc0 = onnx_vals.get("/decoder/generator/noise_convs.0/Conv_output_0")
    onnx_nc1 = onnx_vals.get("/decoder/generator/noise_convs.1/Conv_output_0")
    if onnx_nc0 is not None:
        print(f"ONNX noise_convs.0: shape={onnx_nc0.shape}, range={onnx_nc0.min():.4f} to {onnx_nc0.max():.4f}")
        print(f"  MAE vs PT: {np.abs(nc0.numpy() - onnx_nc0).mean():.4f}")
    if onnx_nc1 is not None:
        print(f"ONNX noise_convs.1: shape={onnx_nc1.shape}, range={onnx_nc1.min():.4f} to {onnx_nc1.max():.4f}")
        print(f"  MAE vs PT: {np.abs(nc1.numpy() - onnx_nc1).mean():.4f}")
