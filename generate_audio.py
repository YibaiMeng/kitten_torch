"""
Generate audio samples comparing three implementations:
  1. ONNX direct (onnxruntime, reference)
  2. kitten-torch (this PyTorch reimplementation)

Saves WAV files to ./audio_samples/

Usage:
  python generate_audio.py                          # downloads model from HuggingFace
  python generate_audio.py --onnx /path/to/model.onnx --voices /path/to/voices.npz
"""
import argparse
import json
import os
import numpy as np
import soundfile as sf
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from kitten_torch import KittenTTS
from kitten_torch.tokenizer import Tokenizer

REPO = "KittenML/kitten-tts-nano-0.8-int8"
SAMPLE_RATE = 24000

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", default=None, help="Path to .onnx model file")
parser.add_argument("--voices", default=None, help="Path to voices.npz")
args = parser.parse_args()

# Download model files (or use provided paths)
config_path = hf_hub_download(repo_id=REPO, filename="config.json")
with open(config_path) as f:
    config = json.load(f)

ONNX_PATH = args.onnx or hf_hub_download(repo_id=REPO, filename=config["model_file"])
VOICES_PATH = args.voices or hf_hub_download(repo_id=REPO, filename=config["voices"])

SPEED_PRIORS = config.get("speed_priors", {})
VOICE_ALIASES = config.get("voice_aliases", {})

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

# ---- ONNX direct session ----
print("Loading ONNX session...")
onnx_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
voices_data = np.load(VOICES_PATH)
tok = Tokenizer()

def run_onnx(text, voice, speed=1.0):
    if voice in SPEED_PRIORS:
        speed = speed * SPEED_PRIORS[voice]
    ids = tok.encode(tts._phonemizer, text)
    input_ids = np.array([ids], dtype=np.int64)
    ref_id = min(len(text), voices_data[voice].shape[0] - 1)
    style_np = voices_data[voice][ref_id:ref_id + 1].astype(np.float32)
    audio = onnx_sess.run(None, {
        "input_ids": input_ids,
        "style": style_np,
        "speed": np.array([speed], dtype=np.float32),
    })[0]
    return audio.astype(np.float32).squeeze()

print(f"\nGenerating {len(SENTENCES)} sentences × {len(VOICES)} voices\n")

for voice in VOICES:
    for i, sentence in enumerate(SENTENCES):
        tag = f"s{i+1}_{voice}"
        print(f"[{tag}] '{sentence[:55]}'" + ("..." if len(sentence) > 55 else ""))

        # 1. ONNX direct (reference)
        onnx_audio = run_onnx(sentence, voice)
        onnx_path = f"audio_samples/{tag}_onnx.wav"
        sf.write(onnx_path, onnx_audio, SAMPLE_RATE)

        # 2. kitten-torch PyTorch (deterministic=True: zero random phase/noise)
        pt_audio = tts.generate(sentence, voice=voice, deterministic=True)
        pt_path = f"audio_samples/{tag}_pytorch.wav"
        sf.write(pt_path, pt_audio, SAMPLE_RATE)

        onnx_max = np.abs(onnx_audio).max()
        pt_max   = np.abs(pt_audio).max()
        print(f"  ONNX:    {len(onnx_audio)/SAMPLE_RATE:.2f}s  max={onnx_max:.3f}  → {onnx_path}")
        print(f"  PyTorch: {len(pt_audio)/SAMPLE_RATE:.2f}s  max={pt_max:.3f}  → {pt_path}")
        if onnx_max > 0:
            print(f"  PT/ONNX ratio: {pt_max/onnx_max:.3f}")
        print()

print("Done! Files saved to audio_samples/")
print("\nAll files:")
for f in sorted(os.listdir("audio_samples")):
    size = os.path.getsize(f"audio_samples/{f}")
    print(f"  audio_samples/{f}  ({size//1024} KB)")
