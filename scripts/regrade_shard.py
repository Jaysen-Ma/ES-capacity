#!/usr/bin/env python
"""Recover a shard whose grading step deadlocked: regrade existing
completions.jsonl with the fixed es_capacity.grade and write records.jsonl +
manifest.json, mirroring es_capacity.generate.run_eval_shards. Skips
re-running vLLM generation entirely.

Usage: python scripts/regrade_shard.py <run_id> <shard_idx> <model_key>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from es_capacity.config import load_config
from es_capacity.datasets import load_minerva_math, minerva_path
from es_capacity.generate import SamplingParamsSpec, shard_dir, write_json, write_jsonl
from es_capacity.grade import grade_batch


def main() -> None:
    run_id, shard_idx, model_key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    machine = os.environ.get("ES_CAPACITY_MACHINE", "gb10")
    cfg = load_config(machine=machine, profile="v1")
    profile_cfg = cfg.section("profile")
    sampling_cfg = cfg.section("sampling")
    grade_cfg = cfg.section("grade")
    seed_cfg = cfg.section("seed")

    run_dir = cfg.runs_dir / run_id
    sdir = shard_dir(run_dir, shard_idx)
    completions_path = sdir / "completions.jsonl"
    completion_rows = [json.loads(l) for l in completions_path.read_text().splitlines() if l]

    num_problems = int(profile_cfg.get("num_problems", -1))
    dataset_path = minerva_path(cfg)
    problems = load_minerva_math(dataset_path, num_problems=None if num_problems < 0 else num_problems)

    sampling = SamplingParamsSpec(
        temperature=float(sampling_cfg.get("temperature", 0.6)),
        top_p=float(sampling_cfg.get("top_p", 0.95)),
        max_new_tokens=int(sampling_cfg.get("max_new_tokens", 4096)),
        template=str(sampling_cfg.get("template", "qwen-boxed")),
        shard_size=int(profile_cfg.get("shard_size", sampling_cfg.get("shard_size", 16))),
    )

    print(f"[regrade] shard {shard_idx:04d} | rows={len(completion_rows)} | problems={len(problems)}")
    t0 = time.time()
    comps_list = [row["completions"] for row in completion_rows]
    graded = grade_batch(
        problems,
        comps_list,
        data_name="minerva_math",
        num_workers=int(grade_cfg.get("num_workers", 16)),
        timeout_sec=float(grade_cfg.get("timeout_sec", 3.0)),
    )
    records = []
    for row, g in zip(completion_rows, graded):
        records.append(
            {
                "idx": row["idx"],
                "preds": g["preds"],
                "scores": g["scores"],
                "c": g["c"],
                "gold": g["gold"],
                "n": len(g["scores"]),
            }
        )
    write_jsonl(sdir / "records.jsonl", records)
    wall = time.time() - t0
    model = cfg.model(model_key)
    write_json(
        sdir / "manifest.json",
        {
            "shard_idx": shard_idx,
            "shard_size": sampling.shard_size,
            "num_problems": len(problems),
            "complete": True,
            "wall_sec": wall,
            "sampling": sampling.fingerprint(),
            "seed_base": int(seed_cfg.get("base", 0)),
            "model_path": str(model.path),
            "note": "regraded from cached completions.jsonl after grade deadlock (timeout=False bug)",
        },
    )
    print(f"[regrade] done shard {shard_idx:04d} wall={wall/60:.1f}m")


if __name__ == "__main__":
    main()
