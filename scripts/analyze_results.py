"""
Given two math_eval.py output directories (base model vs. our ES-trained
model, same benchmark, same n_sampling), compute:
  1. pass@k curves for k=1..n_sampling for both models (unbiased estimator,
     same formula math_eval's own evaluate.py uses for pass@k).
  2. The four-way solvable/unsolvable breakdown (Table 2 style): for each
     question, "solvable" = at least one of the n_sampling completions was
     correct. Cross-tabulate base vs. trained.

Each math_eval.py shard writes `{output_dir}/{benchmark}/{prefix}_s{start}_e{end}.jsonl`
with a `score` field (list[bool], one per sample) already attached by
evaluate(). This script globs all shard files for a run, dedupes by `idx`
(shards cover disjoint question ranges, so this just reassembles the full set).

Usage:
    python analyze_results.py \
        --base-dir EVAL_base/minerva_math \
        --trained-dir EVAL_trained/minerva_math \
        --out-prefix results/minerva_base_vs_trained
"""
import argparse
import glob
import json
import os

import numpy as np


def load_scores(result_dir: str) -> dict:
    """Returns {idx: score_list} across all shard jsonl files in result_dir."""
    files = [f for f in glob.glob(os.path.join(result_dir, "*.jsonl"))]
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {result_dir}")
    by_idx = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                by_idx[sample["idx"]] = sample["score"]
    return by_idx


def pass_at_k_curve(scores_by_idx: dict) -> np.ndarray:
    """pass@k for k=1..N, unbiased estimator, averaged over questions."""
    lens = {len(v) for v in scores_by_idx.values()}
    if len(lens) != 1:
        raise ValueError(f"Inconsistent n_sampling across questions: {lens}")
    n = lens.pop()

    def estimator(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    num_correct = np.array([sum(v) for v in scores_by_idx.values()])
    curve = np.zeros(n)
    for k in range(1, n + 1):
        curve[k - 1] = np.mean([estimator(n, c, k) for c in num_correct])
    return curve


def four_way_breakdown(base: dict, trained: dict) -> dict:
    common_idx = sorted(set(base) & set(trained))
    if len(common_idx) != len(base) or len(common_idx) != len(trained):
        print(
            f"Warning: base has {len(base)} questions, trained has {len(trained)}, "
            f"{len(common_idx)} in common. Using only the common set."
        )

    counts = {"base_solved_trained_solved": 0, "base_solved_trained_failed": 0,
              "base_failed_trained_solved": 0, "base_failed_trained_failed": 0}
    for idx in common_idx:
        b = any(base[idx])
        t = any(trained[idx])
        if b and t:
            counts["base_solved_trained_solved"] += 1
        elif b and not t:
            counts["base_solved_trained_failed"] += 1
        elif not b and t:
            counts["base_failed_trained_solved"] += 1
        else:
            counts["base_failed_trained_failed"] += 1

    n = len(common_idx)
    fractions = {k: round(100 * v / n, 1) for k, v in counts.items()}
    return {"n_questions": n, "counts": counts, "fractions_pct": fractions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="math_eval.py output dir for the base model (.../minerva_math)")
    ap.add_argument("--trained-dir", required=True, help="math_eval.py output dir for the trained model")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--plot", action="store_true", help="Also save a pass@k curve PNG (requires matplotlib)")
    args = ap.parse_args()

    base_scores = load_scores(args.base_dir)
    trained_scores = load_scores(args.trained_dir)

    base_curve = pass_at_k_curve(base_scores)
    trained_curve = pass_at_k_curve(trained_scores)

    breakdown = four_way_breakdown(base_scores, trained_scores)

    out = {
        "n_sampling_base": len(next(iter(base_scores.values()))),
        "n_sampling_trained": len(next(iter(trained_scores.values()))),
        "pass_at_k_base": base_curve.tolist(),
        "pass_at_k_trained": trained_curve.tolist(),
        "four_way_breakdown": breakdown,
    }

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_prefix}_summary.json")

    print("\nFour-way breakdown (base vs. trained), % of questions:")
    for k, v in breakdown["fractions_pct"].items():
        print(f"  {k}: {v}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = np.arange(1, len(base_curve) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(ks, base_curve * 100, label="Base model", marker="o", markersize=3)
        plt.plot(ks, trained_curve * 100, label="ES-trained model", marker="o", markersize=3)
        plt.xscale("log")
        plt.xlabel("k")
        plt.ylabel("pass@k (%)")
        plt.title("Minerva Math: pass@k, base vs. ES-trained")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_passk.png", dpi=150)
        print(f"Wrote {args.out_prefix}_passk.png")


if __name__ == "__main__":
    main()
