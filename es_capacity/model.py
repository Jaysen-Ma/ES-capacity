"""LLM loading and generation helpers.

Backends:
  - ``openai``: Docker / remote vLLM OpenAI-compatible server (recommended)
  - ``vllm``: in-process vLLM ``LLM`` (requires local ``pip install vllm``)
  - ``hf``: HuggingFace ``generate`` (portable fallback)
"""

from __future__ import annotations

from typing import Any, Literal

import torch

Backend = Literal["openai", "vllm", "hf"]


def load_model(model_name: str, backend: Backend = "hf", **kwargs: Any) -> Any:
    """Load a generation engine.

    Returns an opaque engine object consumed by ``generate``.
    """
    if backend == "openai":
        return load_openai(model_name, **kwargs)
    if backend == "vllm":
        return load_vllm(model_name, **kwargs)
    if backend == "hf":
        return load_hf(model_name)
    raise ValueError(f"Unknown backend {backend!r}")


def load_hf(model_name: str) -> dict[str, Any]:
    """Load HF CausalLM + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return {"backend": "hf", "model": model, "tokenizer": tokenizer, "name": model_name}


def load_vllm(
    model_name: str,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Load an in-process vLLM engine for high-throughput sampling."""
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "trust_remote_code": True,
        "enable_prefix_caching": True,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)
    return {"backend": "vllm", "llm": llm, "name": model_name}


def load_openai(
    model_name: str,
    *,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    served_model_name: str | None = None,
    timeout: float = 3600.0,
    max_workers: int = 32,
) -> dict[str, Any]:
    """Connect to a Docker / remote vLLM OpenAI-compatible server.

    ``model_name`` is only a label for logging/checkpoints; the server model id
    is ``served_model_name`` (must match ``--served-model-name`` in Docker).
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    served = served_model_name or model_name
    # Fail fast if the server is down / model missing
    models = client.models.list()
    ids = [m.id for m in models.data]
    if served not in ids:
        raise RuntimeError(
            f"OpenAI server at {base_url} has models {ids}, but served name "
            f"{served!r} was not found. Start Docker with matching "
            f"--served-model-name (see scripts/serve_qwen_vllm_docker.sh)."
        )
    return {
        "backend": "openai",
        "client": client,
        "name": model_name,
        "served_model_name": served,
        "base_url": base_url,
        "max_workers": max_workers,
    }


def apply_noise(model: Any, noise: Any) -> None:
    """Apply a parameter-space perturbation in place."""
    raise NotImplementedError("apply_noise")


def restore_noise(model: Any, noise: Any) -> None:
    """Undo a previously applied parameter-space perturbation in place."""
    raise NotImplementedError("restore_noise")


def generate(
    engine: Any,
    prompts: list[str],
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_new_tokens: int = 4096,
    n: int = 1,
    seed: int | None = None,
    tokenizer: Any = None,
) -> list[list[str]]:
    """Sample ``n`` completions for each prompt.

    Returns a list of length ``len(prompts)``, each an ``n``-list of strings.
    """
    backend = engine.get("backend", "hf") if isinstance(engine, dict) else "hf"
    if backend == "openai":
        return _generate_openai(
            engine,
            prompts,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            n=n,
            seed=seed,
        )
    if backend == "vllm":
        return _generate_vllm(
            engine["llm"],
            prompts,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            n=n,
            seed=seed,
        )
    model = engine["model"] if isinstance(engine, dict) else engine
    tok = tokenizer or (engine.get("tokenizer") if isinstance(engine, dict) else None)
    if tok is None:
        raise ValueError("HF generate requires a tokenizer")
    return _generate_hf(
        model,
        tok,
        prompts,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        n=n,
        seed=seed,
    )


def _generate_openai(
    engine: dict[str, Any],
    prompts: list[str],
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    n: int,
    seed: int | None,
    max_workers: int | None = None,
) -> list[list[str]]:
    """Use ``/v1/completions`` so our full qwen-boxed prompt strings are respected.

    Prompts are submitted concurrently so vLLM can continuous-batch across problems
    (serial submission leaves the GPU underfilled once a single ``n``-sample request
    starts draining).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = engine["client"]
    model = engine["served_model_name"]
    workers = max_workers if max_workers is not None else engine.get("max_workers")
    if workers is None:
        workers = min(32, max(1, len(prompts)))
    workers = max(1, min(int(workers), len(prompts)))

    def _one(i: int, prompt: str) -> list[str]:
        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "temperature": temperature if temperature > 0 else 0.0,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed + i
        resp = client.completions.create(**kwargs)
        choices = sorted(resp.choices, key=lambda c: c.index)
        texts = [c.text for c in choices]
        if len(texts) < n:
            texts = texts + [""] * (n - len(texts))
        return texts[:n]

    if workers == 1 or len(prompts) <= 1:
        return [_one(i, p) for i, p in enumerate(prompts)]

    completions: list[list[str] | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, i, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            i = futs[fut]
            completions[i] = fut.result()
    return [c if c is not None else [] for c in completions]


def _generate_vllm(
    llm: Any,
    prompts: list[str],
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    n: int,
    seed: int | None,
) -> list[list[str]]:
    from vllm import SamplingParams

    params = SamplingParams(
        n=n,
        temperature=temperature if temperature > 0 else 0.0,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    outputs = llm.generate(prompts, params)
    completions: list[list[str]] = []
    for out in outputs:
        texts = [o.text for o in out.outputs]
        if len(texts) < n:
            texts = texts + [""] * (n - len(texts))
        completions.append(texts[:n])
    return completions


@torch.inference_mode()
def _generate_hf(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    n: int,
    seed: int | None,
) -> list[list[str]]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    expanded_prompts: list[str] = []
    for p in prompts:
        expanded_prompts.extend([p] * n)

    inputs = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    outputs = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    completions: list[list[str]] = [[] for _ in prompts]
    for i, seq in enumerate(outputs):
        prompt_idx = i // n
        gen_ids = seq[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        completions[prompt_idx].append(text)
    return completions
