"""
Generate paired A/B blind test clips: ONNX vs PyTorch.

For each phrase + voice, BOTH an ONNX and a PT version are generated.
They are randomly assigned to label A and B.  You listen to both and
pick which one is ONNX — without knowing the assignment.

This is a 2-Alternative Forced Choice (2AFC) paradigm:
every trial has the correct answer by construction, so chance is always 50%.

Usage:
  python gen_ab_test.py          # 20 pairs → 40 clips in ab_test/
  python gen_ab_test.py --n 30
  python gen_ab_test.py --seed 7

Then: python score_ab_test.py
"""
import sys, os, json, random, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf

HF = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF}/kitten_tts_nano_v0_1.onnx"
OUT_DIR = "ab_test"
SR = 24000
FADE_MS = 80  # fade-out length in ms to remove end artifacts


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
    "I cannot believe how quickly time flies.",
    "There is a small cafe on the corner.",
    "The meeting has been moved to Thursday.",
    "Please make sure to lock the door behind you.",
    "She was surprised to see him at the party.",
    "The new policy goes into effect next month.",
    "I would love a cup of coffee right now.",
    "The flowers in the garden smell wonderful.",
    "He decided to take a different route home.",
    "Can you turn the volume down a little?",
    "The library closes at nine in the evening.",
    "I really enjoyed that movie last night.",
    "We have not spoken in almost two years.",
    "The instructions were surprisingly easy to follow.",
    "She prefers tea over coffee in the mornings.",
    "It was the best vacation I have ever had.",
    "The dog sat quietly by the front door.",
    "Would you mind helping me with this?",
    "The sunset painted the sky brilliant orange.",
    "The project deadline is coming up very soon.",
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


def fade_out(audio: np.ndarray, sr: int, ms: int) -> np.ndarray:
    """Apply a linear fade-out over the last `ms` milliseconds."""
    n = int(sr * ms / 1000)
    if n >= len(audio):
        return audio
    result = audio.copy()
    result[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return result


def setup_models(voices_data):
    from kitten_torch import KittenTTS
    from kitten_torch.tokenizer import Tokenizer

    tts = KittenTTS(model_path=ONNX_PATH, voices_path=f"{HF}/voices.npz")
    model = tts._model
    model.eval()

    tok = Tokenizer()
    phonemizer = tts._phonemizer
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    def gen_onnx(text, voice):
        style_np = voices_data[voice][0:1].astype(np.float32)
        ids = tok.encode(phonemizer, text)
        input_ids_np = np.array([ids], dtype=np.int64)
        speed_np = np.array([1.0], dtype=np.float32)
        outs = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed_np})
        audio = outs[0].astype(np.float32)
        return fade_out(audio, SR, FADE_MS)

    def gen_pt(text, voice):
        style_np = voices_data[voice][0:1].astype(np.float32)
        style_pt = torch.from_numpy(style_np)
        ids = tok.encode(phonemizer, text)
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            audio = model(input_ids, style_pt, deterministic=True).squeeze().numpy()
        return fade_out(audio.astype(np.float32), SR, FADE_MS)

    return gen_onnx, gen_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of pairs to generate")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)
    key_path = os.path.join(OUT_DIR, "answer_key.json")

    if os.path.exists(key_path):
        print(f"WARNING: {key_path} already exists.")
        print("Delete it (and the wav files) to regenerate.")
        return

    voices_data = np.load(f"{HF}/voices.npz")
    print("Loading models (ONNX + PyTorch)...")
    gen_onnx, gen_pt = setup_models(voices_data)

    n = args.n
    phrases_pool = (PHRASES * (n // len(PHRASES) + 1))[:n]
    random.shuffle(phrases_pool)
    voices_pool = [random.choice(VOICES) for _ in range(n)]

    pairs = []
    for i, (phrase, voice) in enumerate(zip(phrases_pool, voices_pool)):
        pair_num = i + 1
        # Randomly decide which label (A/B) gets ONNX
        onnx_label = random.choice(["A", "B"])
        pt_label = "B" if onnx_label == "A" else "A"

        print(f"  [{pair_num:3d}/{n}]  {voice}  {phrase[:55]}")

        audio_onnx = gen_onnx(phrase, voice)
        audio_pt   = gen_pt(phrase, voice)

        fname_a = f"pair_{pair_num:03d}_A.wav"
        fname_b = f"pair_{pair_num:03d}_B.wav"

        sf.write(os.path.join(OUT_DIR, fname_a), audio_onnx if onnx_label == "A" else audio_pt, SR)
        sf.write(os.path.join(OUT_DIR, fname_b), audio_onnx if onnx_label == "B" else audio_pt, SR)

        pairs.append({
            "pair": pair_num,
            "file_A": fname_a,
            "file_B": fname_b,
            "onnx_label": onnx_label,   # which file is ONNX (A or B)
            "voice": voice,
            "phrase": phrase,
        })

    with open(key_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\nGenerated {n} pairs ({2*n} files) in {OUT_DIR}/")
    print(f"Answer key → {key_path}  (don't peek!)")
    print()
    print("For each pair, listen to BOTH files (A and B) and decide which is ONNX.")
    print("Then run: python score_ab_test.py")


if __name__ == "__main__":
    main()
