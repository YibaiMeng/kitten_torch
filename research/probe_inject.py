"""
Test: inject ONNX stage2 MRF output into PT conv_post + iSTFT.
If PT produces same audio as ONNX → issue is before conv_post.
If PT still different → issue is IN conv_post/iSTFT.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")
import numpy as np, torch, torch.nn.functional as F, onnx, onnxruntime as ort, re
from onnx import helper, TensorProto
import soundfile as sf

HF = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF}/kitten_tts_nano_v0_1.onnx"
TEXT = "Hello, how are you today?"
VOICE = "expr-voice-2-m"

from kitten_torch import KittenTTS
tts = KittenTTS(model_path=ONNX_PATH, voices_path=f"{HF}/voices.npz")
model = tts._model; model.eval()
voices_data = np.load(f"{HF}/voices.npz")
style_np = voices_data[VOICE][0:1].astype(np.float32)
style_pt = torch.from_numpy(style_np)
ids = tts._phonemize(TEXT)
input_ids = torch.tensor([ids], dtype=torch.long)

import kittentts as _kt, inspect
src = inspect.getsource(_kt.KittenTTS.__init__)
m = re.search(r"enumerate\(list\('(.+?)'\)\)", src, re.DOTALL)
WORD_INDEX = {s: i for i, s in enumerate(list(m.group(1)))}
phonemized = tts._phonemizer.phonemize([TEXT])[0]
normalized = ' '.join(re.findall(r'\w+|[^\w\s]', phonemized))
onnx_ids = [0] + [WORD_INDEX[c] for c in normalized if c in WORD_INDEX] + [0]
input_ids_np = np.array([onnx_ids], dtype=np.int64)

# Get ONNX Div_2 output + noise_res.1 output + audio
model_onnx = onnx.load(ONNX_PATH)
targets = [
    ('/decoder/generator/Div_2_output_0', TensorProto.FLOAT),
    ('/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_output_0', TensorProto.FLOAT),
]
for name, dtype in targets:
    vi = helper.make_tensor_value_info(name, dtype, None)
    model_onnx.graph.output.append(vi)

sess = ort.InferenceSession(model_onnx.SerializeToString(), providers=["CPUExecutionProvider"])
outs = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": np.array([1.0], np.float32)})

onnx_audio = outs[0]
# outs[1] = durations (int64)
onnx_div2 = outs[2].astype(np.float32)    # stage2 MRF output
onnx_conv_post = outs[3].astype(np.float32)  # conv_post output

print(f"ONNX audio max: {np.abs(onnx_audio).max():.4f}")
print(f"ONNX Div_2: shape={onnx_div2.shape} mean={onnx_div2.mean():.4f} std={onnx_div2.std():.4f}")
print(f"ONNX conv_post: shape={onnx_conv_post.shape} mean={onnx_conv_post.mean():.4f} std={onnx_conv_post.std():.4f}")
print(f"  log_mag: mean={onnx_conv_post[:, :11].mean():.4f}  mag_mean={np.exp(onnx_conv_post[:, :11]).mean():.4f}")

# Now: feed ONNX Div_2 through PT's LeakyReLU(0.01) + conv_post + iSTFT
gen = model.decoder.generator
div2_pt = torch.from_numpy(onnx_div2)

with torch.inference_mode():
    x = F.leaky_relu(div2_pt, 0.01)
    x = gen.conv_post(x)
    audio_injected = gen.stft.inverse_stft(x)

pt_conv_post_on_onnx = x.detach()
print(f"\n=== PT conv_post applied to ONNX Div_2 ===")
print(f"PT conv_post: shape={tuple(pt_conv_post_on_onnx.shape)} mean={pt_conv_post_on_onnx.mean():.4f} std={pt_conv_post_on_onnx.std():.4f}")
print(f"  log_mag: mean={pt_conv_post_on_onnx[:,:11].mean():.4f}  mag_mean={pt_conv_post_on_onnx[:,:11].exp().mean():.4f}")
print(f"  Injected audio max: {audio_injected.abs().max():.4f}")

# Compare ONNX conv_post vs PT conv_post (both on ONNX Div_2 input)
T = min(onnx_conv_post.shape[-1], pt_conv_post_on_onnx.shape[-1])
diff = np.abs(onnx_conv_post[..., :T] - pt_conv_post_on_onnx.numpy()[..., :T])
print(f"  ONNX vs PT conv_post diff: max={diff.max():.4f} mean={diff.mean():.4f}")

# Save the injected audio for comparison
sf.write("audio_samples/injected_test.wav", audio_injected.squeeze().numpy(), 24000)
print(f"\nSaved injected test audio to audio_samples/injected_test.wav")

# Also compare with direct PT
with torch.inference_mode():
    audio_pt = model(input_ids, style_pt, deterministic=True)
sf.write("audio_samples/pt_det.wav", audio_pt.squeeze().numpy(), 24000)
print(f"PT deterministic audio max: {audio_pt.abs().max():.4f}")
sf.write("audio_samples/onnx_det.wav", onnx_audio, 24000)
