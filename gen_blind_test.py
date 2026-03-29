"""
Generate blind A/B test clips: ONNX vs PyTorch.
Saves numbered audio files (blind_001.wav, ...) with a hidden answer key.

Usage:
  python gen_blind_test.py          # generates 24 clips in blind_test/
  python gen_blind_test.py --n 30   # generate more clips

After listening to all clips, run:
  python score_blind_test.py
"""
import sys, os, json, random, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf
import re

HF = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF}/kitten_tts_nano_v0_1.onnx"
OUT_DIR = "blind_test"

PHRASES = [
    "The weather is absolutely beautiful today.",
    "Could you please pass me the salt?",
    "I just finished reading a really good novel.",
    "She walked slowly down the empty hallway.",
    "The train arrives at half past seven.",
    "We should probably leave before it gets dark.",
    "Do you have any recommendations for dinner?",
    "He forgot his keys again this morning.",
    "The children were playing in the park.",
    "It takes about twenty minutes to get there.",
    "I can't believe how quickly time flies.",
    "There's a small cafe on the corner.",
    "The meeting has been moved to Thursday.",
    "Please make sure to lock the door behind you.",
    "She was surprised to see him at the party.",
    "The new policy goes into effect next month.",
    "I would love a cup of coffee right now.",
    "They finished the project just before the deadline.",
    "The flowers in the garden smell wonderful.",
    "He decided to take a different route home.",
    "Can you turn the volume down a little?",
    "The library closes at nine in the evening.",
    "I really enjoyed that movie last night.",
    "We haven't spoken in almost two years.",
    "The instructions were surprisingly easy to follow.",
    "She prefers tea over coffee in the mornings.",
    "It was the best vacation I've ever had.",
    "The dog sat quietly by the front door.",
    "Would you mind helping me with this?",
    "The sunset painted the sky brilliant orange.",
]

VOICES = [
    "expr-voice-2-m",
    "expr-voice-2-f",
    "expr-voice-3-m",
    "expr-voice-3-f",
    "expr-voice-4-m",
    "expr-voice-4-f",
    "expr-voice-5-m",
    "expr-voice-5-f",
]


def build_word_index():
    import kittentts as _kt, inspect
    src = inspect.getsource(_kt.KittenTTS.__init__)
    m = re.search(r"enumerate\(list\('(.+?)'\)\)", src, re.DOTALL)
    return {s: i for i, s in enumerate(list(m.group(1)))}


VOCAB_SIZE = 178  # text_encoder embedding has 178 entries (indices 0-177)

def phonemize_to_ids(phonemizer, word_index, text):
    """Shared tokenization for both ONNX and PT. Clamps IDs to valid range."""
    phonemized = phonemizer.phonemize([text])[0]
    normalized = ' '.join(re.findall(r'\w+|[^\w\s]', phonemized))
    ids = [0] + [word_index[c] for c in normalized if c in word_index] + [0]
    return [min(i, VOCAB_SIZE - 1) for i in ids]  # clamp ᵻ (179) and similar


def setup_models(voices_data):
    from kitten_torch import KittenTTS
    tts = KittenTTS(model_path=ONNX_PATH, voices_path=f"{HF}/voices.npz")
    model = tts._model
    model.eval()

    word_index = build_word_index()
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    phonemizer = tts._phonemizer

    def gen_onnx(text, voice):
        style_np = voices_data[voice][0:1].astype(np.float32)
        ids = phonemize_to_ids(phonemizer, word_index, text)
        input_ids_np = np.array([ids], dtype=np.int64)
        speed_np = np.array([1.0], dtype=np.float32)
        outs = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed_np})
        return outs[0]

    def gen_pt(text, voice):
        style_np = voices_data[voice][0:1].astype(np.float32)
        style_pt = torch.from_numpy(style_np)
        ids = phonemize_to_ids(phonemizer, word_index, text)
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            audio = model(input_ids, style_pt, deterministic=True)
        return audio.squeeze().numpy()

    return gen_onnx, gen_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24, help="Number of clips to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    key_path = os.path.join(OUT_DIR, "answer_key.json")

    if os.path.exists(key_path):
        print(f"WARNING: {key_path} already exists. Delete it to regenerate.")
        print("Or look at the answer key and run score_blind_test.py.")
        return

    voices_data = np.load(f"{HF}/voices.npz")
    print("Loading models (ONNX + PyTorch)...")
    gen_onnx, gen_pt = setup_models(voices_data)

    # Build clip list: balanced ONNX/PT, diverse phrases and voices
    n = args.n
    models = (["onnx"] * (n // 2) + ["pt"] * (n - n // 2))
    random.shuffle(models)

    # Sample phrases without repeating until exhausted
    phrases_pool = PHRASES * (n // len(PHRASES) + 1)
    random.shuffle(phrases_pool)
    phrases_pool = phrases_pool[:n]

    voices_pool = [random.choice(VOICES) for _ in range(n)]

    clips = []
    for i, (model_label, phrase, voice) in enumerate(zip(models, phrases_pool, voices_pool)):
        clip_num = i + 1
        fname = f"blind_{clip_num:03d}.wav"
        fpath = os.path.join(OUT_DIR, fname)

        print(f"  [{clip_num:3d}/{n}] {model_label.upper():4s} | {voice} | {phrase[:50]}")
        if model_label == "onnx":
            audio = gen_onnx(phrase, voice)
        else:
            audio = gen_pt(phrase, voice)

        sf.write(fpath, audio, 24000)
        clips.append({
            "file": fname,
            "model": model_label,
            "voice": voice,
            "phrase": phrase,
        })

    # Save answer key (hidden from user until scoring)
    with open(key_path, "w") as f:
        json.dump(clips, f, indent=2)

    print(f"\nGenerated {n} clips in {OUT_DIR}/")
    print(f"Answer key saved to {key_path} — DON'T PEEK until you've guessed all clips!")
    print(f"\nListen to each file and record your guesses (onnx/pt).")
    print(f"Then run: python score_blind_test.py")


if __name__ == "__main__":
    main()
