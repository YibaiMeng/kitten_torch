"""
Generate audio samples comparing ONNX (golden) vs PyTorch implementations.
Saves WAV files to ./audio_samples/
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import os
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort

from kitten_torch import KittenTTS

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"
SAMPLE_RATE = 24000

SENTENCES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis is the art of converting text into spoken audio.",
]

VOICES = ["expr-voice-2-m", "expr-voice-2-f"]

os.makedirs("audio_samples", exist_ok=True)

# ---- Build PT model ----
print("Loading PyTorch model...")
tts = KittenTTS(model_path=ONNX_PATH, voices_path=VOICES_PATH)

# ---- Build ONNX session ----
print("Loading ONNX model...")
onnx_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
voices_data = np.load(VOICES_PATH)

# We need the phonemizer and word index from tts for ONNX too
import kittentts as _kt, inspect, re
src = inspect.getsource(_kt.KittenTTS.__init__)
m = re.search(r"enumerate\(list\('(.+?)'\)\)", src, re.DOTALL)
WORD_INDEX = {s: i for i, s in enumerate(list(m.group(1)))}

def phonemize_to_ids(tts_instance, text):
    phonemized = tts_instance._phonemizer.phonemize([text])[0]
    # Match original kittentts tokenization: normalize whitespace
    normalized = ' '.join(re.findall(r"\w+|[^\w\s]", phonemized))
    ids = [0]  # BOS
    for c in normalized:
        if c in WORD_INDEX:
            ids.append(WORD_INDEX[c])
    ids.append(0)  # EOS
    return ids

def run_onnx(text, voice, speed=1.0):
    ids = phonemize_to_ids(tts, text)
    input_ids = np.array([ids], dtype=np.int64)
    style_np = voices_data[voice][0:1].astype(np.float32)
    speed_arr = np.array([speed], dtype=np.float32)
    audio = onnx_sess.run(None, {
        "input_ids": input_ids,
        "style": style_np,
        "speed": speed_arr,
    })[0]
    return audio.astype(np.float32)

print(f"\nGenerating {len(SENTENCES)} sentences × {len(VOICES)} voices\n")

for voice in VOICES:
    for i, sentence in enumerate(SENTENCES):
        tag = f"s{i+1}_{voice}"
        print(f"[{tag}] '{sentence[:50]}...' " if len(sentence) > 50 else f"[{tag}] '{sentence}'")

        # ONNX golden
        onnx_audio = run_onnx(sentence, voice)
        onnx_path = f"audio_samples/{tag}_onnx.wav"
        sf.write(onnx_path, onnx_audio, SAMPLE_RATE)

        # PyTorch
        pt_audio = tts.generate(sentence, voice=voice, deterministic=False)
        pt_path = f"audio_samples/{tag}_pytorch.wav"
        sf.write(pt_path, pt_audio, SAMPLE_RATE)

        # Stats
        onnx_max = np.abs(onnx_audio).max()
        pt_max = np.abs(pt_audio).max()
        print(f"  ONNX:    {len(onnx_audio)/SAMPLE_RATE:.2f}s  max={onnx_max:.3f}  → {onnx_path}")
        print(f"  PyTorch: {len(pt_audio)/SAMPLE_RATE:.2f}s  max={pt_max:.3f}  → {pt_path}")
        if onnx_max > 0:
            print(f"  PT/ONNX amplitude ratio: {pt_max/onnx_max:.3f}")
        print()

print("Done! Files saved to audio_samples/")
print("\nAll files:")
for f in sorted(os.listdir("audio_samples")):
    size = os.path.getsize(f"audio_samples/{f}")
    print(f"  audio_samples/{f}  ({size//1024} KB)")
