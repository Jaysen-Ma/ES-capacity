#!/usr/bin/env python3
"""Emit the GPQA markdown tables straight from results/gpqa/*.csv.

The summary table in README.md and the full set in docs/gpqa.md are
load-bearing and have been wrong before when copied by hand, so they are
generated instead. Paste the output, or diff it against the committed markdown
to check the tables still match the artifacts.

Usage:
    python scripts/gpqa_tables.py [--dir results/gpqa]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("results/gpqa"))
    args = ap.parse_args()

    summary = {r["arm"]: r for r in read(args.dir / "scores_summary.csv")}
    bias = {r["arm"]: r for r in read(args.dir / "position_bias.csv")}
    pairs = read(args.dir / "pairs_summary.csv")
    order = [a for a in
             ["1.5B-base", "1.5B-ES", "1.5B-RL", "7B-base", "7B-ES", "7B-RL"]
             if a in summary]
    n_seeds = summary[order[0]]["n_seeds"]

    print(f"| Arm | mean acc over {n_seeds} permutations | sd | min | max |")
    print("|---|---|---|---|---|")
    for a in order:
        r = summary[a]
        print(
            f"| {a} | {float(r['acc_mean']) * 100:.2f}% | "
            f"{float(r['acc_std']) * 100:.2f} | {float(r['acc_min']) * 100:.2f}% | "
            f"{float(r['acc_max']) * 100:.2f}% |"
        )

    print()
    print("| Comparison | mean delta (pts) | range | arm ahead on | Wilcoxon p |")
    print("|---|---|---|---|---|")
    for r in pairs:
        print(
            f"| {r['arm']} vs {r['baseline']} | "
            f"**{float(r['delta_mean_pts']):+.2f} ± {float(r['delta_std_pts']):.2f}** | "
            f"{float(r['delta_min_pts']):+.2f} … {float(r['delta_max_pts']):+.2f} | "
            f"{r['seeds_arm_better']}/{r['n_seeds']} | {r['p_wilcoxon_per_question']} |"
        )

    print()
    print("| Arm | picks (A) | (B) | (C) | (D) | spread |")
    print("|---|---|---|---|---|---|")
    for a in order:
        r = bias[a]
        cells = " | ".join(f"{float(r['pct_' + c]):.1f}%" for c in "ABCD")
        print(f"| {a} | {cells} | {float(r['spread_pts']):.1f} pts |")


if __name__ == "__main__":
    main()
