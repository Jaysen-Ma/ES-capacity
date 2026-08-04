"""CLI: aggregate shards into correct_counts + metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from es_capacity.config import load_config
from es_capacity.metrics import aggregate_run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    args = p.parse_args(argv)
    cfg = load_config(machine=args.machine)
    run_dir = Path(args.run_id)
    if not run_dir.is_absolute():
        run_dir = cfg.runs_dir / run_dir
    agg = aggregate_run(run_dir)
    print(f"n={agg['n_total']} problems={agg['num_problems']}")
    for k, v in sorted(agg["passk"].items(), key=lambda x: int(x[0])):
        print(f"  pass@{k}={100*v:.1f}%")


if __name__ == "__main__":
    main()
