"""Sharded vLLM sampling with resume and extensible n."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from es_capacity.config import AppConfig
from es_capacity.datasets import build_prompt, dataset_sha256, load_minerva_math, minerva_path


def request_seed(base_seed: int, shard_idx: int, problem_idx: int) -> int:
    """Deterministic per-request seed: f(base_seed, shard_idx, problem_idx)."""
    h = hashlib.sha256(f"{base_seed}:{shard_idx}:{problem_idx}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**31 - 1)


def git_sha(repo: Path) -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@dataclass
class SamplingParamsSpec:
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 4096
    template: str = "qwen-boxed"
    shard_size: int = 16

    def fingerprint(self) -> dict[str, Any]:
        return asdict(self)


def shard_dir(run_dir: Path, shard_idx: int) -> Path:
    return run_dir / "shards" / f"shard_{shard_idx:04d}"


def shard_complete(run_dir: Path, shard_idx: int) -> bool:
    man = shard_dir(run_dir, shard_idx) / "manifest.json"
    if not man.exists():
        return False
    try:
        return bool(json.loads(man.read_text()).get("complete"))
    except Exception:
        return False


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_run_dir(
    cfg: AppConfig,
    *,
    arm: str,
    model_key: str,
    profile: str,
    run_id: str | None = None,
) -> Path:
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{profile}_{arm}_{ts}"
    run_dir = cfg.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_run_manifest(
    cfg: AppConfig,
    *,
    arm: str,
    model_key: str,
    profile: str,
    sampling: SamplingParamsSpec,
    num_problems: int,
    num_shards: int,
    dataset_path: Path,
) -> dict[str, Any]:
    model = cfg.model(model_key)
    vllm_cfg = cfg.section("vllm")
    try:
        import vllm

        vllm_version = getattr(vllm, "__version__", "unknown")
    except Exception:
        vllm_version = "unavailable"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(cfg.repo_root),
        "config_hash": cfg.hash,
        "arm": arm,
        "profile": profile,
        "model_key": model_key,
        "model_path": str(model.path),
        "model_hf_id": model.hf_id,
        "vllm_version": vllm_version,
        "dtype": vllm_cfg.get("dtype", "bfloat16"),
        "sampling": sampling.fingerprint(),
        "num_problems": num_problems,
        "num_shards": num_shards,
        "n_target": sampling.shard_size * num_shards,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256(dataset_path) if dataset_path.exists() else "",
        "seed_base": cfg.section("seed").get("base", 0),
        "seed_derivation": "sha256(f'{base}:{shard_idx}:{problem_idx}')[:8] % (2**31-1)",
        "template": sampling.template,
    }


class OfflineVLLM:
    """Single offline LLM.generate() engine with prefix caching."""

    def __init__(self, model_path: str | Path, vllm_cfg: dict[str, Any], engine_seed: int = 0):
        from vllm import LLM

        self.model_path = str(model_path)
        kwargs = dict(
            model=self.model_path,
            dtype=vllm_cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.85)),
            max_model_len=int(vllm_cfg.get("max_model_len", 4096)),
            enable_prefix_caching=bool(vllm_cfg.get("enable_prefix_caching", True)),
            tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
            trust_remote_code=True,
            seed=int(engine_seed),
        )
        if vllm_cfg.get("enforce_eager"):
            kwargs["enforce_eager"] = True
        self.llm = LLM(**kwargs)

    def generate_shard(
        self,
        prompts: list[str],
        *,
        problem_indices: list[int],
        shard_idx: int,
        base_seed: int,
        sampling: SamplingParamsSpec,
    ) -> list[dict[str, Any]]:
        from vllm import SamplingParams

        # One SamplingParams per request so each gets a distinct seed.
        # Flat batch: all prompts in one generate() call.
        sp_list = [
            SamplingParams(
                n=sampling.shard_size,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_new_tokens,
                seed=request_seed(base_seed, shard_idx, pidx),
            )
            for pidx in problem_indices
        ]
        outputs = self.llm.generate(prompts, sp_list)
        rows: list[dict[str, Any]] = []
        for out, pidx in zip(outputs, problem_indices):
            texts = [o.text for o in out.outputs]
            finish = [str(o.finish_reason) for o in out.outputs]
            n_tokens = [len(o.token_ids) for o in out.outputs]
            rows.append(
                {
                    "idx": pidx,
                    "completions": texts,
                    "finish_reasons": finish,
                    "num_tokens": n_tokens,
                    "seed": request_seed(base_seed, shard_idx, pidx),
                }
            )
        return rows


def run_eval_shards(
    cfg: AppConfig,
    *,
    arm: str,
    model_key: str | None = None,
    profile: str = "smoke",
    run_id: str | None = None,
    extend_run: str | None = None,
) -> Path:
    """Generate + grade shards for one arm. Returns run_dir."""
    from es_capacity.grade import grade_batch

    profile_cfg = cfg.section("profile")
    sampling_cfg = cfg.section("sampling")
    grade_cfg = cfg.section("grade")
    seed_cfg = cfg.section("seed")

    model_key = model_key or profile_cfg.get("model_key", "base")
    num_problems = int(profile_cfg.get("num_problems", -1))
    shard_size = int(profile_cfg.get("shard_size", sampling_cfg.get("shard_size", 16)))
    num_shards = int(profile_cfg.get("num_shards", 1))

    sampling = SamplingParamsSpec(
        temperature=float(sampling_cfg.get("temperature", 0.6)),
        top_p=float(sampling_cfg.get("top_p", 0.95)),
        max_new_tokens=int(sampling_cfg.get("max_new_tokens", 4096)),
        template=str(sampling_cfg.get("template", "qwen-boxed")),
        shard_size=shard_size,
    )

    dataset_path = minerva_path(cfg)
    problems = load_minerva_math(
        dataset_path,
        num_problems=None if num_problems < 0 else num_problems,
    )
    prompts = [build_prompt(p, sampling.template) for p in problems]
    problem_indices = [int(p["idx"]) for p in problems]

    if extend_run:
        run_dir = cfg.runs_dir / extend_run
        if not run_dir.exists():
            raise FileNotFoundError(f"extend_run not found: {run_dir}")
        man_path = run_dir / "manifest.json"
        existing = json.loads(man_path.read_text()) if man_path.exists() else {}
        # Enforce matching sampling fingerprint
        if existing.get("sampling") and existing["sampling"] != sampling.fingerprint():
            raise ValueError(
                f"Sampling params mismatch vs existing run.\n"
                f"existing={existing.get('sampling')}\nnew={sampling.fingerprint()}"
            )
        existing["num_shards"] = max(int(existing.get("num_shards", 0)), num_shards)
        existing["n_target"] = sampling.shard_size * existing["num_shards"]
        write_json(man_path, existing)
    else:
        run_dir = create_run_dir(cfg, arm=arm, model_key=model_key, profile=profile, run_id=run_id)
        manifest = build_run_manifest(
            cfg,
            arm=arm,
            model_key=model_key,
            profile=profile,
            sampling=sampling,
            num_problems=len(problems),
            num_shards=num_shards,
            dataset_path=dataset_path,
        )
        write_json(run_dir / "manifest.json", manifest)

    model = cfg.model(model_key)
    base_seed = int(seed_cfg.get("base", 0))
    engine = OfflineVLLM(model.path, cfg.section("vllm"), engine_seed=base_seed)

    for sidx in range(num_shards):
        if shard_complete(run_dir, sidx):
            print(f"[skip] shard {sidx:04d} already complete")
            continue
        sdir = shard_dir(run_dir, sidx)
        sdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"[gen] shard {sidx:04d} | problems={len(problems)} | n={shard_size}")
        completion_rows = engine.generate_shard(
            prompts,
            problem_indices=problem_indices,
            shard_idx=sidx,
            base_seed=base_seed,
            sampling=sampling,
        )
        write_jsonl(sdir / "completions.jsonl", completion_rows)

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
        write_json(
            sdir / "manifest.json",
            {
                "shard_idx": sidx,
                "shard_size": shard_size,
                "num_problems": len(problems),
                "complete": True,
                "wall_sec": wall,
                "sampling": sampling.fingerprint(),
                "seed_base": base_seed,
                "model_path": str(model.path),
            },
        )
        print(f"[done] shard {sidx:04d} wall={wall/60:.1f}m")

    # Aggregate after generation
    from es_capacity.metrics.aggregate import aggregate_run

    aggregate_run(run_dir)
    return run_dir
