"""CLI: run sharded evaluation."""

from __future__ import annotations

import argparse

from es_capacity.config import load_config
from es_capacity.generate import run_eval_shards


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Sharded Minerva pass@k evaluation")
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    p.add_argument("--profile", required=True, choices=["smoke", "pilot", "v1", "scale"])
    p.add_argument("--arm", required=True, help="Logical arm name: base|grpo|eggroll|...")
    p.add_argument("--model-key", default=None, help="Override profile model_key")
    p.add_argument("--run-id", default=None)
    p.add_argument("--extend-run", default=None, help="Add shards to an existing run id")
    p.add_argument("--experiment", default=None)
    args = p.parse_args(argv)

    cfg = load_config(machine=args.machine, profile=args.profile, experiment=args.experiment)
    run_dir = run_eval_shards(
        cfg,
        arm=args.arm,
        model_key=args.model_key,
        profile=args.profile,
        run_id=args.run_id,
        extend_run=args.extend_run,
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
