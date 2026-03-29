"""Compare ONNX vs PT decoder intermediate tensors."""
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

# Probe ONNX intermediates
model_proto = onnx.load(ONNX_PATH)
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}
model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)

probe_names = [
    "/decoder/F0_conv/Conv_output_0",
    "/decoder/N_conv/Conv_output_0",
    "/decoder/encode/Mul_output_0",       # encode block output
    "/decoder/decode.0/Mul_output_0",     # decode.0 output
    "/decoder/decode.1/Mul_output_0",     # decode.1 output
    "/decoder/decode.2/Mul_output_0",     # decode.2 output
    "/decoder/decode.3/Mul_output_0",     # decode.3 output (=generator input)
    "/F0_proj/Conv_output_0_Cast_to_float32_output_0",  # F0 predictor output
]
for name in probe_names:
    elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
    vi = onnx.helper.make_tensor_value_info(name, elem_type, None)
    model_copy.graph.output.append(vi)

sess = ort.InferenceSession(model_copy.SerializeToString(), providers=["CPUExecutionProvider"])
results = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed})
output_names = [o.name for o in sess.get_outputs()]
onnx_vals = {}
for name, result in zip(output_names, results):
    if name in set(probe_names):
        arr = np.asarray(result, dtype=np.float32)
        onnx_vals[name] = arr
        print(f"ONNX {name.split('/')[-1]}: shape={arr.shape}, range={arr.min():.4f} to {arr.max():.4f}")

# PT run with hooks
model = build_model(ONNX_PATH)
model.eval()
captured = {}

def make_hook(key):
    def h(m, inp, out):
        captured[key] = out.detach()
    return h

model.decoder.encode.register_forward_hook(make_hook('encode'))
model.decoder.decode[0].register_forward_hook(make_hook('decode0'))
model.decoder.decode[1].register_forward_hook(make_hook('decode1'))
model.decoder.decode[2].register_forward_hook(make_hook('decode2'))
model.decoder.decode[3].register_forward_hook(make_hook('decode3'))
model.decoder.F0_conv.register_forward_hook(make_hook('f0_conv'))
model.decoder.N_conv.register_forward_hook(make_hook('n_conv'))

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

print()
print(f"PT F0_proj output: shape={f0_pred.shape}, range={f0_pred.min():.4f} to {f0_pred.max():.4f}")
for key in ['f0_conv', 'n_conv', 'encode', 'decode0', 'decode1', 'decode2', 'decode3']:
    if key in captured:
        t = captured[key]
        print(f"PT {key}: shape={t.shape}, range={t.min():.4f} to {t.max():.4f}")

print()
# Compare encode output
enc_onnx = onnx_vals.get("/decoder/encode/Mul_output_0")
if enc_onnx is not None and 'encode' in captured:
    pt = captured['encode'].numpy()
    # align shapes
    min_T = min(enc_onnx.shape[-1], pt.shape[-1])
    diff = np.abs(enc_onnx[..., :min_T] - pt[..., :min_T])
    print(f"encode diff: mean={diff.mean():.4f}, max={diff.max():.4f}")

dec3_onnx = onnx_vals.get("/decoder/decode.3/Mul_output_0")
if dec3_onnx is not None and 'decode3' in captured:
    pt = captured['decode3'].numpy()
    min_T = min(dec3_onnx.shape[-1], pt.shape[-1])
    diff = np.abs(dec3_onnx[..., :min_T] - pt[..., :min_T])
    print(f"decode3 diff: mean={diff.mean():.4f}, max={diff.max():.4f}")
