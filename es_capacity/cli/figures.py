"""CLI: produce the three figures from run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from es_capacity.config import load_config
from es_capacity.figures import make_all_figures


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        required=True,
        help="Comma-separated name=run_id pairs, e.g. base=smoke_base_...,grpo=...",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--base-key", default="base")
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    args = p.parse_args(argv)
    cfg = load_config(machine=args.machine)

    arms: dict[str, Path] = {}
    for part in args.runs.split(","):
        name, rid = part.split("=", 1)
        path = Path(rid)
        if not path.is_absolute():
            path = cfg.runs_dir / path
        arms[name.strip()] = path

    out_dir = Path(args.out_dir) if args.out_dir else cfg.runs_dir / "figures"
    make_all_figures(arms, out_dir, base_key=args.base_key)
    print(f"wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
