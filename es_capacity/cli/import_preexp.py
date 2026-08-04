"""Import pre-experiment Minerva base results into sharded run layout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from es_capacity.generate import SamplingParamsSpec, write_json, write_jsonl
from es_capacity.metrics import aggregate_run, passk_curve


def import_preexp_base(
    completions_path: Path,
    metrics_path: Path,
    out_run: Path,
    *,
    shard_size: int = 16,
    arm: str = "base",
) -> Path:
    rows = [json.loads(l) for l in completions_path.open() if l.strip()]
    rows.sort(key=lambda r: int(r["id"]))
    n_samples = int(rows[0]["n_samples"])
    assert n_samples % shard_size == 0, f"n_samples={n_samples} not divisible by {shard_size}"
    num_shards = n_samples // shard_size

    sampling = SamplingParamsSpec(
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=4096,
        template="qwen-boxed",
        shard_size=shard_size,
    )
    out_run.mkdir(parents=True, exist_ok=True)
    write_json(
        out_run / "manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "imported_from": str(completions_path),
            "arm": arm,
            "profile": "v1",
            "model_key": "base",
            "model_path": rows[0].get("model", ""),
            "sampling": sampling.fingerprint(),
            "num_problems": len(rows),
            "num_shards": num_shards,
            "n_target": n_samples,
            "note": "Imported from es-capacity-preexp-outputs; base not re-evaluated.",
            "preexp_metrics": json.loads(metrics_path.read_text()),
        },
    )

    for sidx in range(num_shards):
        sdir = out_run / "shards" / f"shard_{sidx:04d}"
        sdir.mkdir(parents=True, exist_ok=True)
        lo, hi = sidx * shard_size, (sidx + 1) * shard_size
        comps_rows = []
        rec_rows = []
        for r in rows:
            idx = int(r["id"])
            comps = r["completions"][lo:hi]
            scores = [bool(x) for x in r["correct"][lo:hi]]
            preds = list(r.get("extracted", [""] * n_samples)[lo:hi])
            comps_rows.append(
                {
                    "idx": idx,
                    "completions": comps,
                    "finish_reasons": ["imported"] * len(comps),
                    "num_tokens": [None] * len(comps),
                    "seed": None,
                }
            )
            rec_rows.append(
                {
                    "idx": idx,
                    "preds": preds,
                    "scores": scores,
                    "c": int(sum(scores)),
                    "gold": None,
                    "n": len(scores),
                }
            )
        write_jsonl(sdir / "completions.jsonl", comps_rows)
        write_jsonl(sdir / "records.jsonl", rec_rows)
        write_json(
            sdir / "manifest.json",
            {
                "shard_idx": sidx,
                "shard_size": shard_size,
                "num_problems": len(rows),
                "complete": True,
                "wall_sec": 0,
                "sampling": sampling.fingerprint(),
                "imported": True,
                "model_path": rows[0].get("model", ""),
            },
        )

    agg = aggregate_run(out_run)
    print(
        f"imported {out_run} n={agg['n_total']} problems={agg['num_problems']} "
        + " ".join(f"pass@{k}={100*v:.1f}%" for k, v in sorted(((int(k), v) for k, v in agg['passk'].items())))
    )
    return out_run


if __name__ == "__main__":
    import argparse
    import os

    p = argparse.ArgumentParser(
        description=(
            "Import a flat (non-sharded) completions.jsonl + metrics.json produced by an "
            "earlier, ad-hoc eval run into this repo's sharded run layout. Not needed for a "
            "fresh clone: use `es_capacity.cli.eval --arm base` instead unless you already "
            "have legacy outputs to carry over."
        )
    )
    p.add_argument(
        "--preexp-dir",
        default=os.environ.get("PREEXP_DIR"),
        help="Directory containing completions.jsonl + metrics.json (or set $PREEXP_DIR)",
    )
    p.add_argument("--completions", default=None, help="Overrides <preexp-dir>/completions.jsonl")
    p.add_argument("--metrics", default=None, help="Overrides <preexp-dir>/metrics.json")
    p.add_argument("--out-run", required=True, type=Path)
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--arm", default="base")
    args = p.parse_args()

    if not args.completions or not args.metrics:
        if not args.preexp_dir:
            p.error("provide --preexp-dir (or $PREEXP_DIR), or both --completions and --metrics")
    preexp_dir = Path(args.preexp_dir) if args.preexp_dir else None
    completions = Path(args.completions) if args.completions else preexp_dir / "completions.jsonl"
    metrics = Path(args.metrics) if args.metrics else preexp_dir / "metrics.json"

    import_preexp_base(completions, metrics, args.out_run, shard_size=args.shard_size, arm=args.arm)
