"""
Given two math_eval.py output directories (base model vs. our ES-trained
model, same benchmark, same n_sampling), compute:
  1. pass@k curves at k = powers of 2 up to n_sampling (1, 2, 4, ..., n) for
     both models — same k_values convention and same unbiased estimator as
     math_eval's own evaluate.py, and the same sparse-power-of-2 sampling the
     limit-of-RLVR paper's own figures use (not every integer k — the repo
     itself has no plotting code to match against, this replicates the
     paper's figures by eye).
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


def power_of_2_ks(n: int) -> list:
    """1, 2, 4, ..., up to n (same convention as math_eval's evaluate.py and
    the limit-of-RLVR paper's own pass@k figures)."""
    ks = [1]
    power = 1
    while 2 ** power <= n:
        ks.append(2 ** power)
        power += 1
    return ks


def pass_at_k_curve(scores_by_idx: dict) -> tuple:
    """pass@k at k = powers of 2 up to N, unbiased estimator, averaged over
    questions. Returns (ks, values)."""
    lens = {len(v) for v in scores_by_idx.values()}
    if len(lens) != 1:
        raise ValueError(f"Inconsistent n_sampling across questions: {lens}")
    n = lens.pop()

    def estimator(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    num_correct = np.array([sum(v) for v in scores_by_idx.values()])
    ks = power_of_2_ks(n)
    values = [float(np.mean([estimator(n, c, k) for c in num_correct])) for k in ks]
    return ks, values


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
    ap.add_argument("--title", default=None, help="Benchmark display name for the plot title (defaults to out-prefix's basename)")
    ap.add_argument("--plot", action="store_true", help="Also save a pass@k curve PNG (requires matplotlib)")
    args = ap.parse_args()

    base_scores = load_scores(args.base_dir)
    trained_scores = load_scores(args.trained_dir)

    base_ks, base_curve = pass_at_k_curve(base_scores)
    trained_ks, trained_curve = pass_at_k_curve(trained_scores)
    assert base_ks == trained_ks, f"k grids differ: base={base_ks} trained={trained_ks} (n_sampling must match)"

    breakdown = four_way_breakdown(base_scores, trained_scores)

    out = {
        "n_sampling_base": len(next(iter(base_scores.values()))),
        "n_sampling_trained": len(next(iter(trained_scores.values()))),
        "ks": base_ks,
        "pass_at_k_base": base_curve,
        "pass_at_k_trained": trained_curve,
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
        plot_curve(base_ks, base_curve, trained_curve, args.title or os.path.basename(args.out_prefix), f"{args.out_prefix}_passk.png")
        print(f"Wrote {args.out_prefix}_passk.png")


def plot_curve(ks, base_curve, trained_curve, title, out_path):
    """Style matches the limit-of-RLVR paper's own pass@k figures: sparse
    power-of-2 k, log-x with plain-integer tick labels at exactly those k
    values, triangle markers, 0-1 y-axis labeled 'Coverage (pass@k)'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_color = "#1b7a72"    # teal, matches paper's "Base" series
    trained_color = "#e8776a"  # coral, matches paper's "RL" series

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(ks, base_curve, label="Base", marker="^", markersize=5, linewidth=1.8, color=base_color)
    ax.plot(ks, trained_curve, label="ES-trained", marker="^", markersize=5, linewidth=1.8, color=trained_color)

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()

    ymax = max(max(base_curve), max(trained_curve))
    ax.set_ylim(0, min(1.0, max(0.2, np.ceil(ymax * 5) / 5)))
    ax.set_xlabel("Number of Samples $k$")
    ax.set_ylabel("Coverage (pass@$k$)")
    ax.set_title(title)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
