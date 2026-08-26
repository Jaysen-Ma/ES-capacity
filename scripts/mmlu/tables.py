#!/usr/bin/env python3
"""Reduce the MMLU run tree to the table the README carries.

Overall accuracy per arm, plus a paired test of each trained arm against its
own base. Nothing per-subject and nothing per-domain: the question this probe
answers is whether math-only post-training cost anything elsewhere, and the
overall number answers it. Per-subject scores remain in each arm's
results.json if a later question needs them.

Why McNemar and not a two-sample test: every arm answers the identical
questions in the identical order -- MMLU, unlike GPQA, does not shuffle its
choices -- so each question is a matched pair. The test discards the questions
both models get right and both get wrong, and asks whether the trained arm
winning the disagreements is distinguishable from a coin flip. p_adj is
Bonferroni over the four trained-vs-base comparisons.

Prints markdown so the README section can be regenerated rather than
hand-edited, the same rule the GPQA tables follow.

Usage:
    python scripts/mmlu/tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import binomtest

TASK = "mmlu"
ARM_ORDER = ["1.5B-base", "1.5B-ES", "1.5B-RL", "7B-base", "7B-ES", "7B-RL"]


def load(arm_dir: Path) -> dict:
    res = json.loads((arm_dir / "results.json").read_text())
    # n-samples is keyed only by the 57 leaf subjects, so the overall count has
    # to be summed over them. group_subtasks nests one level deep:
    # mmlu -> 4 categories -> 57 subjects.
    counts = {k: v["effective"] for k, v in res["n-samples"].items()}
    members = res.get("group_subtasks", {})

    def n_for(task: str) -> int:
        if task in counts:
            return counts[task]
        return sum(n_for(child) for child in members.get(task, []))

    cfg = res.get("config", {})
    return {
        "acc": res["results"][TASK]["acc,none"],
        "n": n_for(TASK),
        "shots": cfg.get("num_fewshot", "?"),
        "seconds": cfg.get("wall_seconds", "?"),
    }


def correctness(arm_dir: Path) -> dict[tuple[str, int], bool]:
    """Per-question correctness, keyed (subtask, doc_id), from the sample dump."""
    out = {}
    with (arm_dir / "samples.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["subtask"], r["doc_id"])] = bool(r["acc"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-root", type=Path, default=Path("results/mmlu_results"))
    args = ap.parse_args()

    found = {d.name: d for d in args.sweep_root.iterdir()
             if d.is_dir() and (d / "results.json").is_file()}
    arms = [a for a in ARM_ORDER if a in found] + sorted(set(found) - set(ARM_ORDER))
    if not arms:
        raise SystemExit(f"no arms with results.json under {args.sweep_root}")

    data = {a: load(found[a]) for a in arms}
    ok = {a: correctness(found[a]) for a in arms}
    trained = [a for a in arms
               if not a.endswith("-base") and f"{a.split('-')[0]}-base" in ok]

    rows = []
    for arm in arms:
        base = f"{arm.split('-')[0]}-base"
        gained = lost = p_raw = p_adj = ""
        if arm in trained:
            gained = sum(1 for k in ok[arm] if ok[arm][k] and not ok[base][k])
            lost = sum(1 for k in ok[arm] if ok[base][k] and not ok[arm][k])
            p_raw = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
            p_adj = round(min(1.0, p_raw * len(trained)), 6)
            p_raw = round(p_raw, 6)
        rows.append({"arm": arm, "acc": round(data[arm]["acc"], 6), "n": data[arm]["n"],
                     "gained": gained, "lost": lost,
                     "mcnemar_p": p_raw, "mcnemar_p_adj": p_adj,
                     "num_fewshot": data[arm]["shots"],
                     "wall_seconds": data[arm]["seconds"]})

    out = args.sweep_root / "scores.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("| Arm | MMLU 0-shot | McNemar p vs. base |")
    print("|---|---|---|")
    for r in rows:
        q = "—" if r["mcnemar_p_adj"] == "" else f"{r['mcnemar_p_adj']:.2f}"
        print(f"| {r['arm']} | {r['acc'] * 100:.2f}% | {q} |")
    print(f"\nn = {rows[0]['n']} questions, seed 0. wrote {out}")


if __name__ == "__main__":
    main()
