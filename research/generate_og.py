"""Generate audio using the original kittentts package (og reference)."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import os
import numpy as np
import soundfile as sf
import kittentts

SAMPLE_RATE = 24000
HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"

SENTENCES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis is the art of converting text into spoken audio.",
]
VOICES = ["expr-voice-2-m", "expr-voice-2-f"]

os.makedirs("audio_samples", exist_ok=True)

print("Loading original kittentts...")
tts = kittentts.KittenTTS(
    model_path=f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx",
    voices_path=f"{HF_SNAPSHOT}/voices.npz",
)

for voice in VOICES:
    for i, sentence in enumerate(SENTENCES):
        tag = f"s{i+1}_{voice}"
        print(f"[{tag}] '{sentence}'")
        audio = tts.generate(sentence, voice=voice, speed=1.0)
        path = f"audio_samples/{tag}_og.wav"
        sf.write(path, audio, SAMPLE_RATE)
        print(f"  → {path}  ({len(audio)/SAMPLE_RATE:.2f}s, max={np.abs(audio).max():.3f})")

print("\nDone!")
