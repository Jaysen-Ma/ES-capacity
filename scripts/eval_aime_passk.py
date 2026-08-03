"""Evaluate AIME24 pass@k for one or more local HF models.

Paths and defaults come from ``config.toml`` (copy ``config.sample.toml``).

Use the project virtualenv (``paths.venv`` in config)::

    source "$(python -m es_capacity.config venv)/bin/activate"

Recommended (Docker vLLM)::

    bash scripts/serve_qwen_vllm_docker.sh base
    python scripts/eval_aime_passk.py --model-key base

    bash scripts/serve_qwen_vllm_docker.sh instruct
    python scripts/eval_aime_passk.py --model-key instruct

Answer extraction/grading matches Yue et al. (limit-of-RLVR).
Per-problem completions are checkpointed to JSONL so interrupted runs resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from es_capacity.config import load_local_config
from es_capacity.data import build_prompt, load_aime24
from es_capacity.eval.capacity import compare_capacity, format_capacity_table
from es_capacity.eval.passk import passk_from_correct_counts
from es_capacity.model import generate, load_model
from es_capacity.reward import extract_answer, verify

LOG = logging.getLogger("es_capacity.eval_aime")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    if not verbose:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("datasets").setLevel(logging.WARNING)
        logging.getLogger("vllm").setLevel(logging.WARNING)


def _model_slug(model_path: str) -> str:
    return Path(model_path).resolve().name


def _fmt_secs(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _fmt_passk(curve: dict[int, float]) -> str:
    return " ".join(f"pass@{k}={curve[k] * 100:.1f}%" for k in sorted(curve))


def _load_completed_ids(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not jsonl_path.is_file():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[str(row["id"])] = row
    return done


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_complete(row: dict[str, Any] | None, n_samples: int) -> bool:
    return (
        row is not None
        and int(row.get("n_samples", 0)) == n_samples
        and len(row.get("completions", [])) >= n_samples
    )


def eval_one_model(
    model_path: str,
    dataset: list[dict[str, Any]],
    *,
    ks: list[int],
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    backend: str,
    prompt_batch_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    openai_base_url: str,
    openai_model: str | None,
    openai_api_key: str,
    output_dir: Path,
) -> dict[str, Any]:
    slug = _model_slug(model_path)
    model_out = output_dir / slug
    model_out.mkdir(parents=True, exist_ok=True)
    completions_path = model_out / "completions.jsonl"

    LOG.info("Loading model (%s): %s", backend, model_path)
    t_load = time.perf_counter()
    if backend == "openai":
        engine = load_model(
            model_path,
            backend="openai",
            base_url=openai_base_url,
            api_key=openai_api_key,
            served_model_name=openai_model or slug.lower().replace(".", "-"),
            max_workers=prompt_batch_size,
        )
        LOG.info(
            "OpenAI server ready in %s | base_url=%s served=%s max_workers=%d",
            _fmt_secs(time.perf_counter() - t_load),
            engine["base_url"],
            engine["served_model_name"],
            engine.get("max_workers", prompt_batch_size),
        )
    elif backend == "vllm":
        engine = load_model(
            model_path,
            backend="vllm",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        LOG.info("Model ready in %s | slug=%s | backend=vllm", _fmt_secs(time.perf_counter() - t_load), slug)
    else:
        engine = load_model(model_path, backend="hf")
        LOG.info("Model ready in %s | slug=%s | backend=hf", _fmt_secs(time.perf_counter() - t_load), slug)

    done = _load_completed_ids(completions_path)
    pending = [ex for ex in dataset if not _is_complete(done.get(str(ex["id"])), n_samples)]
    LOG.info(
        "Checkpoint %s | done=%d/%d pending=%d | prompt_batch_size=%d n_samples=%d",
        completions_path,
        len(dataset) - len(pending),
        len(dataset),
        len(pending),
        prompt_batch_size,
        n_samples,
    )

    t_model = time.perf_counter()
    finished_this_run = 0
    pbar = tqdm(total=len(pending), desc=slug, dynamic_ncols=True)

    for batch_start in range(0, len(pending), prompt_batch_size):
        batch = pending[batch_start : batch_start + prompt_batch_size]
        prompts = [build_prompt(ex) for ex in batch]
        ids = [str(ex["id"]) for ex in batch]
        LOG.info(
            "Generate batch problems %s (%d prompts × %d samples)",
            ",".join(ids),
            len(batch),
            n_samples,
        )
        t_batch = time.perf_counter()
        completions_batch = generate(
            engine,
            prompts,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            n=n_samples,
            seed=seed + batch_start,
        )
        batch_secs = time.perf_counter() - t_batch
        LOG.info(
            "Batch done in %s (~%s / problem)",
            _fmt_secs(batch_secs),
            _fmt_secs(batch_secs / max(len(batch), 1)),
        )

        for example, comps in zip(batch, completions_batch):
            eid = str(example["id"])
            grades = [verify(example, c) for c in comps]
            preds = [extract_answer(c) for c in comps]
            n_correct = int(sum(grades))
            row = {
                "id": eid,
                "problem": example["problem"],
                "answer": example["answer"],
                "prompt": build_prompt(example),
                "completions": comps,
                "extracted": preds,
                "correct": grades,
                "num_correct": n_correct,
                "n_samples": n_samples,
                "model": model_path,
                "backend": backend,
                "grader": "yue_math",
                "timestamp_utc": _utc_now(),
            }
            _append_jsonl(completions_path, row)
            done[eid] = row
            finished_this_run += 1

            completed_rows = [
                done[str(ex["id"])]
                for ex in dataset
                if _is_complete(done.get(str(ex["id"])), n_samples)
            ]
            running = passk_from_correct_counts(
                [int(r["num_correct"]) for r in completed_rows],
                n_samples,
                ks,
            )
            elapsed_model = time.perf_counter() - t_model
            remaining_n = len(pending) - finished_this_run
            eta = (elapsed_model / finished_this_run) * remaining_n if finished_this_run else 0.0
            pbar.update(1)
            pbar.set_postfix(correct=f"{n_correct}/{n_samples}", eta=_fmt_secs(eta))
            LOG.info(
                "DONE id=%s correct=%d/%d | running(%d): %s | ETA %s",
                eid,
                n_correct,
                n_samples,
                len(completed_rows),
                _fmt_passk(running),
                _fmt_secs(eta),
            )
            LOG.debug("id=%s extracted[:5]=%s", eid, preds[:5])

    pbar.close()

    ordered = []
    for example in dataset:
        eid = str(example["id"])
        if not _is_complete(done.get(eid), n_samples):
            raise RuntimeError(f"Missing completions for problem id={eid}")
        ordered.append(done[eid])

    num_correct = [int(r["num_correct"]) for r in ordered]
    curve = passk_from_correct_counts(num_correct, n_samples, ks)
    LOG.info("Model %s finished in %s | %s", slug, _fmt_secs(time.perf_counter() - t_model), _fmt_passk(curve))
    result = {
        "model": model_path,
        "slug": slug,
        "n_problems": len(dataset),
        "n_samples": n_samples,
        "ks": ks,
        "pass_at_k": {str(k): v for k, v in curve.items()},
        "num_correct_per_problem": num_correct,
        "completions_path": str(completions_path),
        "backend": backend,
        "grader": "yue_math",
        "prompt": "qwen-boxed",
        "gen": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "prompt_batch_size": prompt_batch_size,
        },
        "timestamp_utc": _utc_now(),
    }
    metrics_path = model_out / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    LOG.info("Wrote %s", metrics_path)

    if backend != "openai":
        del engine
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="Path to config.toml (default: ./config.toml)")
    pre_args, remaining = pre.parse_known_args(argv)

    try:
        cfg = load_local_config(pre_args.config, required=True)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    ev = cfg.eval
    vl = cfg.vllm
    default_models = cfg.model_paths()

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[pre],
    )
    p.add_argument(
        "--model-key",
        choices=sorted(cfg.models.keys()),
        default=None,
        help="Use a named model from config.toml (sets --models and --openai-model)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        help="Local HF model directories (default: all models in config.toml)",
    )
    p.add_argument("--ks", nargs="+", type=int, default=list(ev.get("ks", [4, 16, 64])))
    p.add_argument(
        "--n-samples",
        type=int,
        default=int(ev.get("n_samples", 64)),
        help="Samples per problem (must be >= max(ks))",
    )
    p.add_argument("--temperature", type=float, default=float(ev.get("temperature", 0.6)))
    p.add_argument("--top-p", type=float, default=float(ev.get("top_p", 0.95)))
    p.add_argument("--max-new-tokens", type=int, default=int(ev.get("max_new_tokens", 4096)))
    p.add_argument("--seed", type=int, default=int(ev.get("seed", 0)))
    p.add_argument(
        "--backend",
        choices=["openai", "vllm", "hf"],
        default=str(ev.get("backend", "openai")),
        help="Generation backend (default: openai = Docker vLLM server)",
    )
    p.add_argument(
        "--openai-base-url",
        default=str(vl.get("base_url", "http://127.0.0.1:8000/v1")),
        help="vLLM OpenAI base URL when --backend openai",
    )
    p.add_argument(
        "--openai-model",
        default=None,
        help="Served model name on the OpenAI server (Docker --served-model-name)",
    )
    p.add_argument("--openai-api-key", default=str(vl.get("api_key", "EMPTY")))
    p.add_argument(
        "--prompt-batch-size",
        type=int,
        default=int(ev.get("prompt_batch_size", 30)),
        help="How many AIME problems to submit concurrently (also OpenAI client thread pool size)",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(vl.get("gpu_memory_utilization", 0.85)),
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=int(vl["max_model_len"]) if "max_model_len" in vl else None,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(str(ev.get("output_dir", "outputs/aime24_passk"))),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(remaining)
    args.config = pre_args.config

    if args.model_key is not None:
        spec = cfg.model(args.model_key)
        args.models = [str(spec.path)]
        if args.openai_model is None:
            args.openai_model = spec.served_name

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    if args.n_samples < max(args.ks):
        raise SystemExit(f"--n-samples ({args.n_samples}) must be >= max(--ks) ({max(args.ks)})")
    if args.prompt_batch_size < 1:
        raise SystemExit("--prompt-batch-size must be >= 1")
    if args.backend == "hf" and args.prompt_batch_size > 1:
        LOG.warning("HF backend: large prompt batches may OOM; consider --prompt-batch-size 1")
    if args.backend == "openai" and len(args.models) > 1:
        LOG.warning(
            "OpenAI/Docker serves one model at a time. Run with --model-key base, then "
            "restart the container and run --model-key instruct."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(
        "Config | file=%s models=%s backend=%s openai_base_url=%s openai_model=%s ks=%s "
        "n_samples=%d temp=%.2f top_p=%.2f max_new_tokens=%d prompt_batch_size=%d "
        "seed=%d output_dir=%s",
        args.config or "config.toml",
        args.models,
        args.backend,
        args.openai_base_url,
        args.openai_model,
        args.ks,
        args.n_samples,
        args.temperature,
        args.top_p,
        args.max_new_tokens,
        args.prompt_batch_size,
        args.seed,
        args.output_dir,
    )
    LOG.info("Loading AIME24…")
    dataset = load_aime24()
    LOG.info("AIME24 problems: %d | grader=yue_math | prompt=qwen-boxed", len(dataset))

    results: list[dict[str, Any]] = []
    curves: dict[str, dict[int, float]] = {}
    for model_i, model_path in enumerate(args.models, start=1):
        LOG.info("=== Model %d/%d: %s ===", model_i, len(args.models), model_path)
        metrics = eval_one_model(
            model_path,
            dataset,
            ks=args.ks,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            backend=args.backend,
            prompt_batch_size=args.prompt_batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            openai_base_url=args.openai_base_url,
            openai_model=args.openai_model,
            openai_api_key=args.openai_api_key,
            output_dir=args.output_dir,
        )
        results.append(metrics)
        curves[metrics["slug"]] = {int(k): float(v) for k, v in metrics["pass_at_k"].items()}

    summary = compare_capacity(curves, k_max=max(args.ks))
    summary["results"] = results
    summary["benchmark"] = "aime24"
    summary["grader"] = "yue_math"
    summary["timestamp_utc"] = _utc_now()
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOG.info("Final comparison:\n%s", format_capacity_table(summary))
    LOG.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
