"""Compare PT generator with ONNX decoder features fed directly to it."""
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

# Get ONNX intermediates for the generator input
model_proto = onnx.load(ONNX_PATH)
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}
model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)

probe_names = [
    "/decoder/decode.3/Mul_output_0",   # generator features input
    "/decoder/generator/ups.0/ConvTranspose_output_0",  # after ups.0
    "/decoder/generator/ups.1/ConvTranspose_output_0",  # after ups.1
    "/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_output_0",
]
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
for name, result in zip(output_names, results):
    if name in set(probe_names):
        arr = np.asarray(result, dtype=np.float32)
        onnx_vals[name] = arr
        print(f"ONNX {name.split('/')[-1]}: shape={arr.shape}, range={arr.min():.4f} to {arr.max():.4f}")

# PT run with hooks on generator stages
model = build_model(ONNX_PATH)
model.eval()
captured = {}

def make_hook(key):
    def h(m, inp, out):
        captured[key] = out.detach()
    return h

gen = model.decoder.generator
gen.ups[0].register_forward_hook(make_hook('ups0'))
gen.ups[1].register_forward_hook(make_hook('ups1'))
gen.conv_post.register_forward_hook(make_hook('conv_post'))
gen.noise_res[0].register_forward_hook(make_hook('noise_res0'))
gen.noise_res[1].register_forward_hook(make_hook('noise_res1'))
gen.resblocks[0].register_forward_hook(make_hook('rb0'))
gen.resblocks[1].register_forward_hook(make_hook('rb1'))
gen.resblocks[2].register_forward_hook(make_hook('rb2'))
gen.resblocks[3].register_forward_hook(make_hook('rb3'))

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
    audio = gen(features, f0_hz, style, deterministic=True)

print()
for key in ['ups0', 'ups1', 'noise_res0', 'noise_res1', 'rb0', 'rb1', 'rb2', 'rb3', 'conv_post']:
    if key in captured:
        t = captured[key]
        print(f"PT {key}: shape={t.shape}, range={t.min():.4f} to {t.max():.4f}")

print()
# Now feed ONNX decode.3 output into PT generator
onnx_features = onnx_vals.get("/decoder/decode.3/Mul_output_0")
if onnx_features is not None:
    print("=== PT generator with ONNX features ===")
    features_onnx = torch.from_numpy(onnx_features)
    onnx_2T = onnx_features.shape[-1]
    N_onnx = onnx_2T * 300
    # Use ONNX f0 (2T=54 frames)
    f0_onnx_full = torch.from_numpy(onnx_vals.get("/decoder/decode.3/Mul_output_0", onnx_features))  # placeholder
    # Actually we need ONNX F0 at 2T resolution for sine generator
    # Let's just use a simple test: pass zeros f0 (voiced=False)
    f0_for_gen = torch.zeros(1, 1, N_onnx)
    captured2 = {}
    def make_hook2(key):
        def h(m, inp, out):
            captured2[key] = out.detach()
        return h
    gen.conv_post.register_forward_hook(make_hook2('cp'))
    with torch.no_grad():
        audio2 = gen(features_onnx, f0_for_gen, style, deterministic=True)
    print(f"Audio (ONNX features, zero f0): range={audio2.min():.4f} to {audio2.max():.4f}")
    if 'cp' in captured2:
        cp = captured2['cp']
        lm = cp[:, :11, :]
        print(f"conv_post log_mag: range={lm.min():.4f} to {lm.max():.4f}, max_exp={lm.max().exp():.4f}")
