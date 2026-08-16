#!/usr/bin/env python3
"""Extract the derived record from an `es-at-scale` trainer stdout log.

The 7B runs used `--logging none`, so the trainer's stdout is the only
per-iteration record that exists for them. This turns that stdout into two
committed artifacts, so the multi-megabyte raw logs never need to be in git:

  <out>/training_curves.csv   per-iteration reward mean/std/min/max + wall time
  <out>/inloop_eval.csv       the trainer's own eval-suite pass@1, per task

Usage:
    python scripts/parse_training_log.py --out-root results/ \
        --run 7b-sigma001-iter50=results/logs/qwen7b-math-sigma001.log
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

# `Namespace(task='math', model_name='Qwen/Qwen2.5-7B', ...)` — the trainer echoes
# its parsed args as the second line of stdout.
RE_NAMESPACE = re.compile(r"^Namespace\((.*)\)\s*$")
# `=== Epoch 1; Iteration 7 ===`
RE_ITER_START = re.compile(r"^=== Epoch (\d+); Iteration (\d+) ===\s*$")
# `=== Epoch 1; Iteration 7 finished in 229.99 ===`
RE_ITER_END = re.compile(r"^=== Epoch (\d+); Iteration (\d+) finished in ([\d.]+) ===\s*$")
# `Mean reward: 0.4569, std: 0.0501, min: 0.3242, max: 0.5898`
RE_REWARD = re.compile(
    r"^Mean reward: ([\d.eE+-]+), std: ([\d.eE+-]+), min: ([\d.eE+-]+), max: ([\d.eE+-]+)\s*$"
)
# `aime -- eval pass@1: 0.1666 --`
RE_EVAL = re.compile(r"^(\w+) -- eval pass@1: ([\d.eE+-]+) --\s*$")


def parse_args_line(line: str) -> dict:
    """Recover the trainer's argparse Namespace as a plain dict."""
    m = RE_NAMESPACE.match(line)
    if not m:
        return {}
    # `Namespace(...)` is not itself literal_eval-able (it's a call), but each of
    # its keyword values is a literal, so parse the call and eval the values.
    call = ast.parse(f"f({m.group(1)})", mode="eval").body
    return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}


def parse_log(path: Path) -> tuple[dict, list[dict], list[dict]]:
    config: dict = {}
    iters: dict[int, dict] = {}
    evals: list[dict] = []
    current: int | None = None

    with path.open(errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")

            if not config and line.startswith("Namespace("):
                config = parse_args_line(line)
                continue

            if m := RE_ITER_END.match(line):
                it = int(m.group(2))
                iters.setdefault(it, {"iteration": it})["wall_s"] = float(m.group(3))
                current = None
                continue

            if m := RE_ITER_START.match(line):
                current = int(m.group(2))
                iters.setdefault(current, {"iteration": current})["epoch"] = int(m.group(1))
                continue

            if m := RE_REWARD.match(line):
                if current is None:
                    continue
                row = iters.setdefault(current, {"iteration": current})
                row["reward_mean"] = float(m.group(1))
                row["reward_std"] = float(m.group(2))
                row["reward_min"] = float(m.group(3))
                row["reward_max"] = float(m.group(4))
                continue

            if m := RE_EVAL.match(line):
                # An eval block either precedes iteration 1 (the trainer's baseline
                # pass over the starting weights — for a resumed run, that's the
                # checkpoint it resumed from) or runs inside the iteration whose
                # update it is scoring. Label the two cases distinctly rather than
                # collapsing them into one ambiguous iteration number.
                evals.append(
                    {
                        "eval_point": "baseline" if current is None else f"iter{current}",
                        "task": m.group(1),
                        "pass@1": float(m.group(2)),
                    }
                )

    # An interrupted run leaves a final started-but-unfinished iteration with no
    # reward line; drop it rather than emitting a half-row.
    rows = [r for it, r in sorted(iters.items()) if "reward_mean" in r]
    return config, rows, evals


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=LOG",
        help="Run name and its log path; repeat once per run.",
    )
    args = ap.parse_args()

    for spec in args.run:
        name, _, log_path = spec.partition("=")
        log = Path(log_path)
        config, rows, evals = parse_log(log)
        out = args.out_root / name

        write_csv(
            out / "training_curves.csv",
            rows,
            ["iteration", "epoch", "reward_mean", "reward_std", "reward_min", "reward_max", "wall_s"],
        )
        if evals:
            write_csv(out / "inloop_eval.csv", evals, ["eval_point", "task", "pass@1"])

        span = f"{rows[0]['iteration']}-{rows[-1]['iteration']}" if rows else "none"
        print(
            f"{name}: {len(rows)} iterations ({span}), {len(evals)} eval rows"
            f"  [sigma={config.get('sigma')} alpha={config.get('alpha')} "
            f"bs={config.get('batch_size')} pop={config.get('population_size')}]"
        )


if __name__ == "__main__":
    main()
