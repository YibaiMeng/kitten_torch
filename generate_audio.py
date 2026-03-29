"""
Generate audio samples comparing ONNX (golden) vs PyTorch implementations.
Saves WAV files to ./audio_samples/

Usage:
  python generate_audio.py                          # downloads model from HuggingFace
  python generate_audio.py --onnx /path/to/model.onnx --voices /path/to/voices.npz
"""
import argparse
import os
import numpy as np
import soundfile as sf
import onnxruntime as ort

from kitten_torch import KittenTTS
from kitten_torch.tokenizer import Tokenizer

SAMPLE_RATE = 24000

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", default=None, help="Path to .onnx model file")
parser.add_argument("--voices", default=None, help="Path to voices.npz")
args = parser.parse_args()
ONNX_PATH = args.onnx
VOICES_PATH = args.voices

SENTENCES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis is the art of converting text into spoken audio.",
]

VOICES = ["expr-voice-2-m", "expr-voice-2-f"]

os.makedirs("audio_samples", exist_ok=True)

# ---- Build PT model (downloads from HuggingFace if paths not given) ----
print("Loading PyTorch model...")
tts = KittenTTS(model_path=ONNX_PATH, voices_path=VOICES_PATH)

# Resolve actual paths after possible HF download
_onnx = ONNX_PATH or tts._model  # tts already loaded; grab paths for ONNX session
# Re-derive ONNX path if not specified
if ONNX_PATH is None:
    from huggingface_hub import hf_hub_download
    ONNX_PATH = hf_hub_download("KittenML/kitten-tts-nano-0.1", "kitten_tts_nano_v0_1.onnx")
if VOICES_PATH is None:
    from huggingface_hub import hf_hub_download
    VOICES_PATH = hf_hub_download("KittenML/kitten-tts-nano-0.1", "voices.npz")

# ---- Build ONNX session ----
print("Loading ONNX model...")
onnx_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
voices_data = np.load(VOICES_PATH)

tok = Tokenizer()

def run_onnx(text, voice, speed=1.0):
    ids = tok.encode(tts._phonemizer, text)
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
