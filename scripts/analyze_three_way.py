"""
Three-way comparison: base model vs. ES-trained vs. an RL-trained checkpoint
(e.g. SimpleRL-Zoo), same benchmark, same n_sampling for all three.

Produces:
  1. A 3-line pass@k plot (Base / ES-trained / RL), same power-of-2-k style
     as analyze_results.py's plot_curve.
  2. Two four-way solvable/unsolvable tables — Base-vs-ES and Base-vs-RL,
     side by side — so the narrowing/gain rates of the two post-training
     methods can be compared directly against the same base model.

Reuses analyze_results.py's load_scores / pass_at_k_curve / four_way_breakdown
so all three scripts agree on the exact same estimator and score-loading logic.

Usage:
    python analyze_three_way.py \
        --base-dir results/iter50/base/aime24 \
        --es-dir results/iter50/trained/aime24 \
        --rl-dir results/iter50/rl/aime24 \
        --out-prefix results/iter50/aime24_threeway \
        --title AIME24 \
        --plot
"""
import argparse
import json
import os

import numpy as np

from analyze_results import load_scores, pass_at_k_curve, four_way_breakdown


def plot_three_way(base_ks, base_curve, es_ks, es_curve, rl_ks, rl_curve, title, out_path):
    """Each series is plotted over its own k-range — rl_ks may be a shorter
    prefix of base_ks/es_ks (e.g. a cheap low-k RL check before committing to
    the full budget), in which case its line simply stops early while base/ES
    continue, rather than forcing all three onto one truncated range."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Base": "#1b7a72",       # teal
        "ES-trained": "#e8776a",  # coral
        "RL (SimpleRL-Zoo)": "#6a5acd",  # slate purple
    }

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for label, ks, curve in [
        ("Base", base_ks, base_curve),
        ("ES-trained", es_ks, es_curve),
        ("RL (SimpleRL-Zoo)", rl_ks, rl_curve),
    ]:
        ax.plot(ks, curve, label=label, marker="^", markersize=5, linewidth=1.8, color=colors[label])

    all_ks = sorted(set(base_ks) | set(es_ks) | set(rl_ks))
    ax.set_xscale("log")
    ax.set_xticks(all_ks)
    ax.set_xticklabels([str(k) for k in all_ks])
    ax.minorticks_off()

    ymax = max(max(base_curve), max(es_curve), max(rl_curve))
    ax.set_ylim(0, min(1.0, max(0.2, np.ceil(ymax * 5) / 5)))
    ax.set_xlabel("Number of Samples $k$")
    ax.set_ylabel("Coverage (pass@$k$)")
    ax.set_title(title)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--es-dir", required=True)
    ap.add_argument("--rl-dir", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    base_scores = load_scores(args.base_dir)
    es_scores = load_scores(args.es_dir)
    rl_scores = load_scores(args.rl_dir)

    base_ks, base_curve = pass_at_k_curve(base_scores)
    es_ks, es_curve = pass_at_k_curve(es_scores)
    rl_ks, rl_curve = pass_at_k_curve(rl_scores)
    assert base_ks == es_ks, f"base/ES k grids differ: base={base_ks} es={es_ks} (these should always match)"
    if rl_ks != base_ks:
        prefix_len = min(len(rl_ks), len(base_ks))
        assert rl_ks == base_ks[:prefix_len], (
            f"RL's k grid isn't a prefix of base/ES's: base={base_ks} rl={rl_ks}. "
            "RL must use the same n_sampling as base/ES, or a smaller uniform override."
        )
        print(f"Note: RL n_sampling ({rl_ks[-1]}) is smaller than base/ES ({base_ks[-1]}) — "
              f"RL's pass@k curve and 'solvable' bar only cover k up to {rl_ks[-1]}.")

    n_base = len(next(iter(base_scores.values())))
    n_rl = len(next(iter(rl_scores.values())))
    es_breakdown = four_way_breakdown(base_scores, es_scores)
    rl_breakdown = four_way_breakdown(base_scores, rl_scores)

    out = {
        "n_sampling_base_es": n_base,
        "n_sampling_rl": n_rl,
        "ks_base_es": base_ks,
        "ks_rl": rl_ks,
        "pass_at_k_base": base_curve,
        "pass_at_k_es": es_curve,
        "pass_at_k_rl": rl_curve,
        "four_way_breakdown_es_vs_base": es_breakdown,
        "four_way_breakdown_rl_vs_base": rl_breakdown,
        "note": (
            None if n_rl == n_base else
            f"'solvable' means different sample budgets for ES ({n_base}) vs RL ({n_rl}) — "
            "not a perfectly apples-to-apples threshold between the two tables."
        ),
    }

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_prefix}_summary.json")

    print(f"\nES vs base (solvable within {n_base} samples), % of questions:")
    for k, v in es_breakdown["fractions_pct"].items():
        print(f"  {k}: {v}%")
    print(f"\nRL vs base (solvable within {n_rl} samples), % of questions:")
    for k, v in rl_breakdown["fractions_pct"].items():
        print(f"  {k}: {v}%")

    if args.plot:
        title = args.title or os.path.basename(args.out_prefix)
        plot_three_way(base_ks, base_curve, es_ks, es_curve, rl_ks, rl_curve, title, f"{args.out_prefix}_passk.png")
        print(f"Wrote {args.out_prefix}_passk.png")


if __name__ == "__main__":
    main()
