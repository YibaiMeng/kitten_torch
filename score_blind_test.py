"""
Score the blind A/B test.

Usage:
  python score_blind_test.py

Reads answer_key.json from blind_test/, prompts for guesses,
then prints accuracy + binomial test p-value.

You can also pass guesses as a string:
  python score_blind_test.py --guesses "oopoopppoopoopppoooo"
  (o = onnx, p = pytorch, in clip order)
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

from scipy import stats

OUT_DIR = "blind_test"


def load_key():
    key_path = os.path.join(OUT_DIR, "answer_key.json")
    if not os.path.exists(key_path):
        print(f"ERROR: {key_path} not found. Run gen_blind_test.py first.")
        sys.exit(1)
    with open(key_path) as f:
        return json.load(f)


def prompt_guesses(clips):
    guesses = []
    print(f"\n{'Clip':<10} {'File':<20} {'Your guess (o=onnx / p=pt)'}")
    print("-" * 55)
    for i, clip in enumerate(clips):
        while True:
            raw = input(f"  {i+1:3d}/{len(clips)}  {clip['file']:<20}  [o/p]: ").strip().lower()
            if raw in ("o", "onnx"):
                guesses.append("onnx")
                break
            elif raw in ("p", "pt", "pytorch"):
                guesses.append("pt")
                break
            else:
                print("    Please enter 'o' for ONNX or 'p' for PyTorch.")
    return guesses


def parse_guess_string(s):
    result = []
    for c in s.lower():
        if c == 'o':
            result.append("onnx")
        elif c == 'p':
            result.append("pt")
    return result


def score(clips, guesses):
    n = len(clips)
    correct = sum(g == c["model"] for g, c in zip(guesses, clips))
    wrong = n - correct

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  Clips:   {n}")
    print(f"  Correct: {correct}  ({100*correct/n:.1f}%)")
    print(f"  Wrong:   {wrong}  ({100*wrong/n:.1f}%)")

    # Two-sided binomial test (H0: p=0.5 chance level)
    result = stats.binomtest(correct, n, p=0.5, alternative='two-sided')
    p_val = result.pvalue

    print(f"\n  Binomial test (H0: guessing at chance, p=0.5):")
    print(f"    p-value = {p_val:.4f}")
    if p_val < 0.001:
        sig = "*** (p < 0.001) — highly significant, you can hear a difference"
    elif p_val < 0.01:
        sig = "**  (p < 0.01)  — very significant"
    elif p_val < 0.05:
        sig = "*   (p < 0.05)  — significant, you can likely tell them apart"
    elif p_val < 0.10:
        sig = "~   (p < 0.10)  — marginal trend"
    else:
        sig = "    (p >= 0.10) — not significant, models sound the same to you"
    print(f"    {sig}")

    # Break down by model
    onnx_clips = [(g, c) for g, c in zip(guesses, clips) if c["model"] == "onnx"]
    pt_clips   = [(g, c) for g, c in zip(guesses, clips) if c["model"] == "pt"]
    onnx_correct = sum(g == c["model"] for g, c in onnx_clips)
    pt_correct   = sum(g == c["model"] for g, c in pt_clips)

    print(f"\n  Accuracy by model:")
    print(f"    ONNX clips ({len(onnx_clips)} total): {onnx_correct} correct ({100*onnx_correct/max(1,len(onnx_clips)):.0f}%)")
    print(f"    PT   clips ({len(pt_clips)} total):   {pt_correct} correct ({100*pt_correct/max(1,len(pt_clips)):.0f}%)")

    # Show per-clip breakdown
    print(f"\n  {'Clip':<10} {'File':<22} {'Answer':<8} {'Guess':<8} {'✓/✗'}")
    print("  " + "-" * 58)
    for i, (g, c) in enumerate(zip(guesses, clips)):
        mark = "✓" if g == c["model"] else "✗"
        print(f"  {i+1:3d}       {c['file']:<22} {c['model']:<8} {g:<8} {mark}  {c['phrase'][:30]}")

    print()
    return p_val, correct, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guesses", type=str, default=None,
                        help="Guess string: o=onnx, p=pt, e.g. 'ooppooppoo...'")
    args = parser.parse_args()

    clips = load_key()

    if args.guesses:
        guesses = parse_guess_string(args.guesses)
        if len(guesses) != len(clips):
            print(f"ERROR: got {len(guesses)} guesses but {len(clips)} clips.")
            sys.exit(1)
    else:
        print(f"\nFound {len(clips)} clips in {OUT_DIR}/")
        print("Enter your guess for each clip (o = ONNX, p = PyTorch).")
        guesses = prompt_guesses(clips)

    score(clips, guesses)


if __name__ == "__main__":
    main()
