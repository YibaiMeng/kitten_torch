"""
Score the paired A/B blind test.

Usage:
  python score_ab_test.py                    # interactive prompt
  python score_ab_test.py --guesses AABBBA…  # one char per pair: A or B (which you think is ONNX)
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

from scipy import stats

OUT_DIR = "ab_test"


def load_key():
    key_path = os.path.join(OUT_DIR, "answer_key.json")
    if not os.path.exists(key_path):
        print(f"ERROR: {key_path} not found. Run gen_ab_test.py first.")
        sys.exit(1)
    with open(key_path) as f:
        return json.load(f)


def prompt_guesses(pairs):
    guesses = []
    print(f"\n{'Pair':<6} {'Files':<32} {'Which is ONNX? [A/B]'}")
    print("-" * 60)
    for p in pairs:
        while True:
            raw = input(f"  {p['pair']:3d}   {p['file_A']}  {p['file_B']}   [A/B]: ").strip().upper()
            if raw in ("A", "B"):
                guesses.append(raw)
                break
            print("    Please enter A or B.")
    return guesses


def score(pairs, guesses):
    n = len(pairs)
    correct = sum(g == p["onnx_label"] for g, p in zip(guesses, pairs))
    wrong = n - correct

    result = stats.binomtest(correct, n, p=0.5, alternative="two-sided")
    p_val = result.pvalue

    print("\n" + "=" * 60)
    print("  A/B TEST RESULTS")
    print("=" * 60)
    print(f"\n  Pairs:   {n}")
    print(f"  Correct: {correct}  ({100*correct/n:.1f}%)")
    print(f"  Wrong:   {wrong}  ({100*wrong/n:.1f}%)")
    print(f"\n  Binomial test (H0: chance = 50%):")
    print(f"    p-value = {p_val:.4f}")

    if p_val < 0.001:
        sig = "*** p < 0.001  — you can clearly hear a difference"
    elif p_val < 0.01:
        sig = "**  p < 0.01   — very likely audible difference"
    elif p_val < 0.05:
        sig = "*   p < 0.05   — significant, models sound different"
    elif p_val < 0.10:
        sig = "~   p < 0.10   — marginal, maybe a small difference"
    else:
        sig = "    p >= 0.10  — not significant, they sound the same"
    print(f"    {sig}")

    # Needed for significance
    from math import ceil
    from scipy.stats import binom
    for target_p in (0.10, 0.05, 0.01):
        # find smallest k where P(X >= k | n, 0.5) <= target_p/2
        for k in range(n // 2, n + 1):
            if binom.sf(k - 1, n, 0.5) <= target_p / 2:
                print(f"    Need {k}/{n} correct for p < {target_p}")
                break

    print(f"\n  {'Pair':<6} {'ONNX was':<10} {'Your guess':<12} {'✓/✗':<4} Phrase")
    print("  " + "-" * 70)
    for g, p in zip(guesses, pairs):
        mark = "✓" if g == p["onnx_label"] else "✗"
        print(f"  {p['pair']:3d}   {p['onnx_label']:<10} {g:<12} {mark}    {p['phrase'][:40]}")

    print()
    return p_val, correct, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guesses", type=str, default=None,
                        help="Guess string, one char per pair: A or B (which you think is ONNX)")
    args = parser.parse_args()

    pairs = load_key()

    if args.guesses:
        guesses = [c.upper() for c in args.guesses if c.upper() in ("A", "B")]
        if len(guesses) != len(pairs):
            print(f"ERROR: {len(guesses)} guesses but {len(pairs)} pairs.")
            sys.exit(1)
    else:
        print(f"\nFound {len(pairs)} pairs in {OUT_DIR}/")
        print("For each pair, listen to both A and B, then enter which one sounds like ONNX.")
        guesses = prompt_guesses(pairs)

    score(pairs, guesses)


if __name__ == "__main__":
    main()
