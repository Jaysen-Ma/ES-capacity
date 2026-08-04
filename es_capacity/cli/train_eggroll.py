"""CLI: validate / train EGGROLL (gated on rising reward)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from es_capacity.config import load_config
from es_capacity.posttrain.eggroll import EggrollConfig, EggrollPerturber
from es_capacity.posttrain.loop import ESLoop
from es_capacity.posttrain.probe_multilora import probe_multilora
from es_capacity.posttrain.scorer import YueMathScorer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    p.add_argument("--experiment", default="minerva_7b")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--validate-1p5b", action="store_true", help="Require rising reward on 1.5B")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    cfg = load_config(machine=args.machine, experiment=args.experiment)
    model_1p5 = cfg.model("base_1p5b")
    probe_out = cfg.runs_dir / "probes" / "multilora_sm121.json"
    result = probe_multilora(str(model_1p5.path), out_path=probe_out)
    print("Multi-LoRA probe:", json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit("Multi-LoRA probe FAILED — do not start EGGROLL training")

    if args.probe_only:
        return

    egg = cfg.section("eggroll")
    val = egg.get("validate", {}) if isinstance(egg.get("validate"), dict) else cfg.raw.get("eggroll", {}).get("validate", {})
    # load_config merges experiment; validate subsection may be under eggroll.validate
    # From minerva_7b.toml [eggroll.validate] becomes nested under eggroll if TOML nests —
    # our flat loader puts it as raw['eggroll'] and raw may not nest validate.
    # Read from experiment file keys we defined:
    validate_cfg = {
        "population_size": int(egg.get("population_size", 32) if not args.validate_1p5b else 32),
        "num_iterations": 5 if args.validate_1p5b else int(egg.get("num_iterations", 100)),
        "lora_r": int(egg.get("lora_r", 1)),
        "sigma": float(egg.get("sigma", 0.001)),
        "learning_rate": float(egg.get("learning_rate", 0.0002)),
    }
    # Prefer dedicated validate block if present via remapping
    # (toml [eggroll.validate] -> eggroll_validate in some loaders; ours deep-merges nested)
    nested = cfg.raw.get("eggroll", {})
    if "validate" in nested and isinstance(nested["validate"], dict):
        nested_v = nested["validate"]
        if args.validate_1p5b:
            validate_cfg["population_size"] = int(nested_v.get("population_size", 32))
            validate_cfg["num_iterations"] = int(nested_v.get("num_iterations", 30))

    out = Path(args.out_dir) if args.out_dir else cfg.runs_dir / "train" / ("eggroll_1p5b_validate" if args.validate_1p5b else "eggroll_7b")
    pert = EggrollPerturber(
        EggrollConfig(
            population_size=validate_cfg["population_size"],
            lora_r=validate_cfg["lora_r"],
            sigma=validate_cfg["sigma"],
            learning_rate=validate_cfg["learning_rate"],
        )
    )
    loop = ESLoop(pert, YueMathScorer(), output_dir=out)
    loop.save_meta()
    print(
        f"EGGROLL skeleton ready at {out}. "
        "Full evaluate/update wiring needs Multi-LoRA engine attach "
        "(probe passed). Next: implement engine-backed evaluate in a follow-up step "
        "or run third_party/eggroll/es_lora_multinode.py with single-GPU flags."
    )
    # Write a launch helper script
    launch = out / "launch_hint.sh"
    launch.write_text(
        f"""#!/bin/bash
# Single-GPU EGGROLL launch hint (adapt from third_party/eggroll)
# Probe result: {probe_out}
# After reward curve rises on 1.5B, scale to 7B.
cd "$(dirname "$0")/../../.."
# ray start --head --num-gpus=1
# python third_party/eggroll/es_lora_multinode.py \\
#   --model-name {model_1p5.path} \\
#   --population-size {validate_cfg['population_size']} \\
#   --lora-r {validate_cfg['lora_r']} \\
#   --sigma {validate_cfg['sigma']} \\
#   --learning-rate {validate_cfg['learning_rate']} \\
#   --task math:math12k \\
#   --num-iterations {validate_cfg['num_iterations']} \\
#   --prompt-batch-size 8 --max-tokens 1024
"""
    )
    print(f"wrote {launch}")


if __name__ == "__main__":
    main()
