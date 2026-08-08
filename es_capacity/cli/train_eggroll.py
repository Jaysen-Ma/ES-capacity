"""CLI: train EGGROLL, resumable and safe to leave running overnight.

## State model

ES noise is regenerated from `(base_seed, pop_step, pair_idx, layer_idx)`, so the
entire training trajectory is reconstructible from the scalar fitness differences.
`records.jsonl` is therefore the real checkpoint (a few hundred KB); a materialized
15 GB HF checkpoint is only produced when something needs to *load* the model --
i.e. for evaluation, or to shorten a later resume.

## Intervening in a running job

- `touch <run_dir>/STOP` -- finishes the current step, saves a checkpoint, exits 0.
- `SIGTERM` / `SIGINT` (Ctrl-C, `kill <pid>`) -- same graceful path, no half-written
  checkpoint and no lost steps. A second signal aborts immediately.
- `<run_dir>/heartbeat.json` -- updated at every phase change, so progress and
  liveness can be checked without reading the vLLM log.

Nothing here waits forever: the weight-update RPC is bounded by
`[eggroll].update_timeout_sec`, and a stalled generation shows up as a heartbeat
whose `updated_utc` stops advancing.
"""

from __future__ import annotations

import argparse
import json
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from es_capacity.config import load_config
from es_capacity.posttrain.eggroll import EggrollConfig, EggrollPerturber, EngineConfig
from es_capacity.posttrain.loop import ESLoop
from es_capacity.posttrain.probe_multilora import probe_multilora
from es_capacity.posttrain.scorer import YueMathScorer

_ABORT = {"requested": False, "hard": False}


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        if _ABORT["requested"]:
            _ABORT["hard"] = True
            raise KeyboardInterrupt(f"second signal {signum}; aborting now")
        _ABORT["requested"] = True
        print(
            f"\n[signal {signum}] finishing the current step, then saving and exiting. "
            "Send again to abort immediately.",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _heartbeat(run_dir: Path, **fields) -> None:
    payload = {"updated_utc": datetime.now(timezone.utc).isoformat(), **fields}
    tmp = run_dir / "heartbeat.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(run_dir / "heartbeat.json")


def _data_cursor(perm: list[int], step: int, batch_size: int) -> list[int]:
    n = len(perm)
    start = (step * batch_size) % n
    return [perm[(start + i) % n] for i in range(batch_size)]


def _load_records(run_dir: Path) -> list[tuple[int, int, float]]:
    path = run_dir / "records.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.open():
        if line.strip():
            r = json.loads(line)
            out.append((int(r["pop_step"]), int(r["pair_idx"]), float(r["fitness_diff"])))
    return out


def _append_records(run_dir: Path, step: int, records: list[tuple[int, int, float]]) -> None:
    with (run_dir / "records.jsonl").open("a") as f:
        for pop_step, pair_idx, diff in records:
            f.write(
                json.dumps(
                    {"step": step, "pop_step": pop_step, "pair_idx": pair_idx, "fitness_diff": diff}
                )
                + "\n"
            )


def _build_eggroll_config(egg: dict, *, args: argparse.Namespace) -> EggrollConfig:
    engine_raw = egg.get("engine", {}) if isinstance(egg.get("engine"), dict) else {}
    engine = EngineConfig(
        max_model_len=int(args.max_model_len or engine_raw.get("max_model_len", 6144)),
        max_num_seqs=int(args.max_num_seqs or engine_raw.get("max_num_seqs", 256)),
        max_loras=int(args.max_loras or engine_raw.get("max_loras", 16)),
        gpu_memory_utilization=float(engine_raw.get("gpu_memory_utilization", 0.85)),
        dtype=str(engine_raw.get("dtype", "bfloat16")),
    )
    return EggrollConfig(
        population_size=int(args.population_size or egg.get("population_size", 64)),
        lora_r=int(egg.get("lora_r", 1)),
        sigma=float(args.sigma or egg.get("sigma", 0.001)),
        learning_rate=float(args.learning_rate or egg.get("learning_rate", 0.001)),
        fitness_shaping=str(egg.get("fitness_shaping", "centered_rank")),
        base_seed=int(args.base_seed),
        prompt_batch_size=int(args.prompt_batch_size or egg.get("prompt_batch_size", 16)),
        max_tokens=int(args.max_tokens or egg.get("max_tokens", 4096)),
        temperature=float(egg.get("temperature", 0.0)),
        steps_per_adapter=int(args.steps_per_adapter or egg.get("steps_per_adapter", 1)),
        save_freq=int(args.save_every or egg.get("save_freq", 20)),
        checkpoint_keep=int(args.checkpoint_keep or egg.get("checkpoint_keep", 20)),
        train_dataset=str(egg.get("train_dataset", "math_lvl3to5")),
        update_timeout_sec=float(egg.get("update_timeout_sec", 900.0)),
        engine=engine,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    p.add_argument("--experiment", default="minerva_7b")
    p.add_argument("--model-key", default="base", help="Base model to train (config models.<key>)")
    p.add_argument("--run-id", default="eggroll_7b", help="Subdirectory under runs/train/")
    p.add_argument("--num-iterations", type=int, required=True, help="ES steps to run this invocation")
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--prompt-batch-size", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--steps-per-adapter", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-loras", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None, help="Materialize an HF checkpoint every N steps")
    p.add_argument("--checkpoint-keep", type=int, default=None, help="How many checkpoints to retain on disk")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--fresh", action="store_true", help="Ignore existing state; start the run over")
    p.add_argument("--skip-probe", action="store_true", help="Skip the Multi-LoRA go/no-go probe")
    args = p.parse_args(argv)

    cfg = load_config(machine=args.machine, experiment=args.experiment)
    _install_signal_handlers()

    if not args.skip_probe:
        probe_out = cfg.runs_dir / "probes" / "multilora_sm121.json"
        result = probe_multilora(str(cfg.model("base_1p5b").path), out_path=probe_out)
        if not result["ok"]:
            raise SystemExit(f"Multi-LoRA probe FAILED -- not starting training: {result}")
        print(f"Multi-LoRA probe OK ({probe_out})")

    egg = cfg.section("eggroll")
    egg["engine"] = cfg.section("eggroll", "engine")
    egg_cfg = _build_eggroll_config(egg, args=args)

    run_dir = cfg.runs_dir / "train" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").unlink(missing_ok=True)
    state_path = run_dir / "state.json"

    start_step = 0
    resume_ckpt_dir: Path | None = None
    prior_records: list[tuple[int, int, float]] = []
    if state_path.exists() and not args.fresh:
        state = json.loads(state_path.read_text())
        if int(state.get("base_seed", args.base_seed)) != args.base_seed:
            raise SystemExit(
                f"base_seed mismatch on resume: state has {state.get('base_seed')}, "
                f"got --base-seed {args.base_seed}"
            )
        start_step = int(state["step"])
        ck = state.get("ckpt_dir")
        resume_ckpt_dir = Path(ck) if ck else None
        all_records = _load_records(run_dir)
        # Records already baked into the materialized checkpoint must not be replayed.
        replay_from = int(state.get("ckpt_records", 0))
        prior_records = all_records[replay_from:]
        print(
            f"Resuming {args.run_id} at step {start_step} "
            f"(ckpt={resume_ckpt_dir}, replaying {len(prior_records)} records)"
        )
    else:
        for f in ("records.jsonl", "history.jsonl", "state.json"):
            (run_dir / f).unlink(missing_ok=True)
        print(f"Starting fresh run {args.run_id}")

    model = cfg.model(args.model_key)
    model_cfg = json.loads((model.path / "config.json").read_text())

    from es_capacity.datasets import load_simplerl_math

    problems = load_simplerl_math(split=egg_cfg.train_dataset, cfg=cfg)
    perm = list(range(len(problems)))
    random.Random(egg_cfg.base_seed).shuffle(perm)

    perturber = EggrollPerturber(egg_cfg)
    perturber.attach_run(
        base_model_path=model.path,
        model_cfg=model_cfg,
        run_dir=run_dir,
        resume_ckpt_dir=resume_ckpt_dir,
    )
    perturber.all_records = _load_records(run_dir)

    grade_cfg = cfg.section("grade")
    scorer = YueMathScorer(
        data_name="minerva_math",
        num_workers=int(grade_cfg.get("num_workers", 16)),
        timeout_sec=float(grade_cfg.get("timeout_sec", 3.0)),
    )
    loop = ESLoop(perturber, scorer, output_dir=run_dir)
    loop.load_history()
    loop.save_meta()

    end_step = start_step + args.num_iterations
    print(
        f"Plan: steps [{start_step}, {end_step}) | pop={egg_cfg.population_size} "
        f"prompt_batch={egg_cfg.prompt_batch_size} max_tokens={egg_cfg.max_tokens} "
        f"sigma={egg_cfg.sigma} lr={egg_cfg.learning_rate} "
        f"steps_per_adapter={egg_cfg.steps_per_adapter} save_every={egg_cfg.save_freq}",
        flush=True,
    )
    print(f"To stop gracefully: touch {run_dir / 'STOP'}", flush=True)
    _heartbeat(run_dir, phase="starting_engine", step=start_step, end_step=end_step)

    perturber.ensure_engine()
    if prior_records:
        _heartbeat(run_dir, phase="replaying_records", step=start_step, n_records=len(prior_records))
        t0 = time.time()
        perturber.replay_records(prior_records)
        print(f"Replayed {len(prior_records)} records in {time.time()-t0:.1f}s", flush=True)

    stop_reason = "completed"
    last_step = start_step - 1
    try:
        for step in range(start_step, end_step):
            if _ABORT["requested"]:
                stop_reason = "signal"
                break
            if (run_dir / "STOP").exists():
                stop_reason = "STOP file"
                break

            _heartbeat(run_dir, phase="generating", step=step, end_step=end_step)
            idxs = _data_cursor(perm, step, egg_cfg.prompt_batch_size)
            batch = [problems[i] for i in idxs]
            t_step = time.time()
            result = loop.step(step, [b["prompt"] for b in batch], [str(b["answer"]) for b in batch])
            wall = time.time() - t_step
            last_step = step

            new_records = perturber.all_records[-perturber.num_pairs :]
            _append_records(run_dir, step, new_records)
            d = perturber.last_diag
            print(
                f"[step {step:04d}] reward={result['mean_reward']:.4f} "
                f"std={d.get('fitness_std', 0):.4f} distinct={d.get('mean_distinct_frac', 0):.2f} "
                f"tok_mean={d.get('tok_mean', 0):.0f} trunc={d.get('truncated_frac', 0):.1%} "
                f"gen={d.get('gen_wall_sec', 0):.0f}s grade={d.get('grade_wall_sec', 0):.0f}s "
                f"upd={d.get('update_wall_sec', 0):.0f}s tot={wall:.0f}s",
                flush=True,
            )
            _write_state(run_dir, step + 1, egg_cfg.base_seed, resume_ckpt_dir, perturber)
            _heartbeat(
                run_dir, phase="stepped", step=step + 1, end_step=end_step,
                mean_reward=result["mean_reward"], step_wall_sec=round(wall, 1), **d,
            )

            if egg_cfg.save_freq > 0 and (step + 1) % egg_cfg.save_freq == 0:
                _heartbeat(run_dir, phase="saving_checkpoint", step=step + 1)
                ck = perturber.save_checkpoint(step + 1)
                resume_ckpt_dir = ck
                _write_state(run_dir, step + 1, egg_cfg.base_seed, ck, perturber, baked=True)
                print(f"  saved checkpoint {ck}", flush=True)
    except KeyboardInterrupt:
        stop_reason = "hard abort"
        raise
    finally:
        if last_step >= start_step and stop_reason != "hard abort":
            _heartbeat(run_dir, phase="final_checkpoint", step=last_step + 1)
            ck = perturber.save_checkpoint(last_step + 1)
            _write_state(run_dir, last_step + 1, egg_cfg.base_seed, ck, perturber, baked=True)
            print(f"Final checkpoint: {ck}", flush=True)
        perturber.close()
        _heartbeat(run_dir, phase="exited", step=last_step + 1, stop_reason=stop_reason)
        print(f"Stopped ({stop_reason}) at step {last_step + 1}.", flush=True)


def _write_state(
    run_dir: Path, next_step: int, base_seed: int, ckpt_dir, perturber, *, baked: bool = False
) -> None:
    prev = {}
    if (run_dir / "state.json").exists():
        prev = json.loads((run_dir / "state.json").read_text())
    state = {
        "step": next_step,
        "base_seed": base_seed,
        "ckpt_dir": str(ckpt_dir) if ckpt_dir else None,
        # How many records are already folded into ckpt_dir; the rest get replayed.
        "ckpt_records": len(perturber.all_records) if baked else prev.get("ckpt_records", 0),
    }
    (run_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
