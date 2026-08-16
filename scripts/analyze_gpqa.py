#!/usr/bin/env python3
"""Reduce an `lm_eval` GPQA-diamond sweep to the derived artifacts we commit.

The raw sweep tree is gitignored: its `samples_*.jsonl` are ~2.6 MB per arm and
embed the verbatim GPQA questions, including presigned S3 URLs that trip GitHub
secret scanning. Everything the README's claims rest on is derived here instead:

  <out>/scores.csv        per-arm accuracy, n correct, stderr, model path
  <out>/per_question.csv  doc_id x arm correctness matrix (+ doc_hash, so a
                          re-run can verify it scored the same shuffled items)
  <out>/mcnemar.csv       exact paired test for each arm against its base

Usage:
    python scripts/analyze_gpqa.py --sweep-root results/gpqa_results \
        --out results/gpqa --pair 7B-ES:7B-base
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TASK = "gpqa_diamond_zeroshot"


def mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar (binomial sign test on the discordant pairs)."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) * 0.5**n
    return min(1.0, 2.0 * tail)


def load_arm(arm_dir: Path) -> tuple[dict, dict[int, int], dict[int, str]]:
    """Return (results_json, {doc_id: correct}, {doc_id: doc_hash}) for one arm."""
    results_paths = sorted(arm_dir.rglob("results_*.json"))
    if not results_paths:
        raise FileNotFoundError(f"no results_*.json under {arm_dir}")
    results = json.loads(results_paths[-1].read_text())

    correct: dict[int, int] = {}
    hashes: dict[int, str] = {}
    for samples_path in sorted(arm_dir.rglob(f"samples_{TASK}_*.jsonl")):
        with samples_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                correct[rec["doc_id"]] = int(rec["acc"])
                hashes[rec["doc_id"]] = rec.get("doc_hash", "")
    return results, correct, hashes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="ARM:BASELINE",
        help="Paired comparison to run; repeat once per pair.",
    )
    args = ap.parse_args()

    arms = sorted(d for d in args.sweep_root.iterdir() if d.is_dir())
    loaded = {d.name: load_arm(d) for d in arms}
    args.out.mkdir(parents=True, exist_ok=True)

    # --- scores.csv -------------------------------------------------------
    score_rows = []
    for name, (results, correct, _) in loaded.items():
        metrics = results["results"][TASK]
        n_total = results["n-samples"][TASK]["effective"]
        score_rows.append(
            {
                "arm": name,
                "acc": round(metrics["acc,none"], 6),
                "n_correct": round(metrics["acc,none"] * n_total),
                "n_total": n_total,
                "acc_stderr": round(metrics["acc_stderr,none"], 6),
                "has_per_question": bool(correct),
                "model_path": results["config"]["model_args"]["pretrained"],
            }
        )
    with (args.out / "scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(score_rows[0]))
        w.writeheader()
        w.writerows(sorted(score_rows, key=lambda r: r["arm"]))

    # --- per_question.csv -------------------------------------------------
    with_samples = [n for n, (_, c, _) in loaded.items() if c]
    doc_ids = sorted(loaded[with_samples[0]][1])
    hashes = loaded[with_samples[0]][2]
    for name in with_samples[1:]:
        other = loaded[name][2]
        mismatched = [d for d in doc_ids if other.get(d) != hashes.get(d)]
        if mismatched:
            raise SystemExit(
                f"{name} scored different documents than {with_samples[0]} "
                f"at doc_ids {mismatched[:5]} — arms are not comparable."
            )
    with (args.out / "per_question.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc_id", "doc_hash", *with_samples])
        for d in doc_ids:
            w.writerow([d, hashes[d], *(loaded[n][1][d] for n in with_samples)])

    # --- mcnemar.csv ------------------------------------------------------
    mc_rows = []
    for spec in args.pair:
        arm, _, baseline = spec.partition(":")
        for side in (arm, baseline):
            if not loaded[side][1]:
                print(f"skipping {spec}: {side} has no samples_*.jsonl in the sweep")
                break
        else:
            a, b = loaded[arm][1], loaded[baseline][1]
            gained = sum(1 for d in doc_ids if a[d] and not b[d])
            lost = sum(1 for d in doc_ids if b[d] and not a[d])
            mc_rows.append(
                {
                    "arm": arm,
                    "baseline": baseline,
                    "acc_arm": round(sum(a.values()) / len(doc_ids), 6),
                    "acc_baseline": round(sum(b.values()) / len(doc_ids), 6),
                    "delta_pts": round(100 * (sum(a.values()) - sum(b.values())) / len(doc_ids), 2),
                    "gained": gained,
                    "lost": lost,
                    "p_mcnemar_exact": round(mcnemar_exact(gained, lost), 4),
                }
            )
    if mc_rows:
        with (args.out / "mcnemar.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(mc_rows[0]))
            w.writeheader()
            w.writerows(mc_rows)

    for r in sorted(score_rows, key=lambda r: r["arm"]):
        flag = "" if r["has_per_question"] else "  (scores only, no samples)"
        print(f"{r['arm']:<24} {r['acc']*100:5.1f}%  {r['n_correct']:>3}/{r['n_total']}{flag}")
    print()
    for r in mc_rows:
        print(
            f"{r['arm']:<24} vs {r['baseline']:<10} {r['delta_pts']:+6.2f} pts   "
            f"+{r['gained']} / -{r['lost']}   p={r['p_mcnemar_exact']}"
        )


if __name__ == "__main__":
    main()
