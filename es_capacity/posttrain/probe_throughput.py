"""Measure ES rollout throughput under vLLM Multi-LoRA.

An EGGROLL step generates `population x prompt_batch` sequences, where each
population member is a distinct rank-r LoRA adapter. Step wall-time therefore
depends on two things this probe measures directly:

1. How much throughput Multi-LoRA costs versus plain generation, as a function
   of the number of *distinct* adapters batched together.
2. The absolute tokens/s at the concurrency a real ES step produces.

Both are hardware-specific, so run this on whatever machine will do the
training (single GPU or rented multi-GPU) and budget from its JSON output
rather than from numbers measured elsewhere.

Adapters are synthesised from the model's config.json — no weights are loaded
on CPU and no PEFT training is required — using EGGROLL's noise scaling so
generated sequence lengths stay realistic.
"""

from __future__ import annotations

import json
import math
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _module_shapes(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Return {module: (in_features, out_features)} for one decoder layer."""
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", hidden // n_heads))
    q_out = n_heads * head_dim
    kv_out = n_kv * head_dim
    return {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }


def synth_lora_adapters(
    model_path: str | Path,
    out_dir: Path,
    *,
    num_adapters: int,
    rank: int = 1,
    sigma: float = 0.001,
    seed: int = 0,
) -> list[Path]:
    """Write `num_adapters` PEFT-format rank-r adapters, EGGROLL-scaled."""
    import torch
    from safetensors.torch import save_file

    cfg = json.loads((Path(model_path) / "config.json").read_text())
    shapes = _module_shapes(cfg)
    n_layers = int(cfg["num_hidden_layers"])
    std_a = math.sqrt(sigma)
    std_b = math.sqrt(sigma / rank)

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": list(TARGET_MODULES),
        "base_model_name_or_path": str(model_path),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    paths: list[Path] = []
    for idx in range(num_adapters):
        tensors: dict[str, Any] = {}
        for layer in range(n_layers):
            for mod, (in_f, out_f) in shapes.items():
                block = "self_attn" if mod.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
                stem = f"base_model.model.model.layers.{layer}.{block}.{mod}"
                a = torch.normal(0.0, std_a, size=(rank, in_f), generator=gen)
                b = torch.normal(0.0, std_b, size=(out_f, rank), generator=gen)
                tensors[f"{stem}.lora_A.weight"] = a.to(torch.bfloat16)
                tensors[f"{stem}.lora_B.weight"] = b.to(torch.bfloat16)
        apath = out_dir / f"adapter_{idx:05d}"
        apath.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(apath / "adapter_model.safetensors"))
        (apath / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
        paths.append(apath)
    return paths


@dataclass
class ProbeResult:
    num_adapters: int
    distinct_adapters: int
    concurrency: int
    max_tokens: int
    lora_rank: int
    wall_sec: float
    output_tokens: int
    prompt_tokens: int
    tokens_per_sec: float
    mean_output_len: float
    derate_vs_no_lora: float | None = None


def _load_prompts(num: int, template: str = "qwen-boxed") -> list[str]:
    from es_capacity.datasets import build_prompt, load_minerva_math

    problems = load_minerva_math()
    prompts = [build_prompt(p, template) for p in problems]
    if not prompts:
        raise RuntimeError("no prompts available")
    return [prompts[i % len(prompts)] for i in range(num)]


def run_probe(
    model_path: str,
    *,
    adapter_specs: list[tuple[int, int]],
    concurrency: int,
    max_tokens: int,
    lora_rank: int,
    vllm_cfg: dict[str, Any],
    adapter_root: Path,
    es_step_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Sweep adapter configurations at fixed concurrency, then optionally time
    one realistic ES step (population x prompt_batch).

    Each spec is (max_loras, distinct_adapters). Holding `max_loras` fixed while
    varying `distinct_adapters` separates the cost of *having* LoRA slots from
    the cost of batching many *different* adapters in one forward pass.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    configs: list[tuple[int, int, int]] = [(n, d, concurrency) for n, d in adapter_specs]
    if es_step_shape is not None:
        pop, prompt_batch = es_step_shape
        configs.append((pop, pop, pop * prompt_batch))

    results: list[ProbeResult] = []
    baseline_tps: float | None = None

    for num_adapters, distinct, conc in configs:
        prompts = _load_prompts(conc)
        adapters: list[Path] = []
        if num_adapters > 0:
            adir = adapter_root / f"n{distinct}_r{lora_rank}"
            if adir.exists():
                shutil.rmtree(adir)
            t_synth = time.time()
            adapters = synth_lora_adapters(
                model_path, adir, num_adapters=distinct, rank=lora_rank
            )
            print(f"[probe] synthesised {distinct} adapters in {time.time()-t_synth:.1f}s", flush=True)

        kwargs: dict[str, Any] = dict(
            model=model_path,
            dtype=vllm_cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.85)),
            max_model_len=int(vllm_cfg.get("max_model_len", 4096)),
            enable_prefix_caching=bool(vllm_cfg.get("enable_prefix_caching", True)),
            tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
            max_num_seqs=conc,
            trust_remote_code=True,
            seed=0,
        )
        if num_adapters > 0:
            kwargs.update(enable_lora=True, max_loras=num_adapters, max_lora_rank=lora_rank)

        print(
            f"[probe] engine: max_loras={num_adapters} distinct={distinct} concurrency={conc}",
            flush=True,
        )
        llm = LLM(**kwargs)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        lora_reqs = (
            [
                LoRARequest(f"es_{i % distinct}", (i % distinct) + 1, str(adapters[i % distinct]))
                for i in range(conc)
            ]
            if num_adapters > 0
            else None
        )

        t0 = time.time()
        outs = llm.generate(prompts, sp, lora_request=lora_reqs)
        wall = time.time() - t0

        out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
        in_tok = sum(len(o.prompt_token_ids or []) for o in outs)
        tps = out_tok / wall if wall else 0.0
        if num_adapters == 0:
            baseline_tps = tps
        res = ProbeResult(
            num_adapters=num_adapters,
            distinct_adapters=distinct,
            concurrency=conc,
            max_tokens=max_tokens,
            lora_rank=lora_rank if num_adapters else 0,
            wall_sec=round(wall, 1),
            output_tokens=out_tok,
            prompt_tokens=in_tok,
            tokens_per_sec=round(tps, 1),
            mean_output_len=round(out_tok / len(outs), 1),
            derate_vs_no_lora=round(tps / baseline_tps, 3) if baseline_tps else None,
        )
        results.append(res)
        print(f"[probe] -> {res.tokens_per_sec} tok/s  ({res.wall_sec}s, {out_tok} output tokens)", flush=True)

        del llm
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        if adapters:
            shutil.rmtree(adapters[0].parent, ignore_errors=True)

    return {
        "model": model_path,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "gpu": _gpu_name(),
            "num_gpus": _num_gpus(),
        },
        "vllm_config": {k: v for k, v in vllm_cfg.items() if k in ("dtype", "max_model_len", "gpu_memory_utilization", "tensor_parallel_size")},
        "results": [asdict(r) for r in results],
    }


def _gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def _num_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> None:
    import argparse

    from es_capacity.config import load_config

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    p.add_argument("--model", default=None, help="Model path; defaults to config models.<--model-key>")
    p.add_argument("--model-key", default="base")
    p.add_argument(
        "--adapters",
        default="0,64,256",
        help=(
            "Comma-separated adapter configs. 'N' means max_loras=N with N distinct "
            "adapters; 'N:D' pins max_loras=N while batching only D distinct adapters "
            "(isolates slot cost from multi-adapter batching cost); '0' is the no-LoRA baseline"
        ),
    )
    p.add_argument("--concurrency", type=int, default=1024, help="Sequences in flight for the sweep")
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--lora-rank", type=int, default=1)
    p.add_argument(
        "--es-step",
        default="256x16",
        help="Also time one realistic ES step as <population>x<prompt_batch>; empty to skip",
    )
    p.add_argument("--adapter-root", default="/dev/shm/es_capacity_probe_adapters")
    p.add_argument("--out", default=None, help="Default: <runs_dir>/probes/multilora_throughput.json")
    args = p.parse_args(argv)

    cfg = load_config(machine=args.machine)
    model = args.model or str(cfg.model(args.model_key).path)
    out = Path(args.out) if args.out else cfg.runs_dir / "probes" / "multilora_throughput.json"

    es_shape = None
    if args.es_step:
        pop, batch = args.es_step.lower().split("x")
        es_shape = (int(pop), int(batch))

    specs: list[tuple[int, int]] = []
    for token in args.adapters.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            slots, distinct = token.split(":", 1)
            specs.append((int(slots), int(distinct)))
        else:
            specs.append((int(token), int(token)))

    report = run_probe(
        model,
        adapter_specs=specs,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        lora_rank=args.lora_rank,
        vllm_cfg=cfg.section("vllm"),
        adapter_root=Path(args.adapter_root),
        es_step_shape=es_shape,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
