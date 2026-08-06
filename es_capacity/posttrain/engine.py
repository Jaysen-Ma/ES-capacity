"""vLLM Multi-LoRA training I/O: adapter materialization, generation, and the
in-place ES weight update.

Kept separate from eggroll.py so the ES math (paper-derived, MIT) stays free of
vLLM/safetensors plumbing. Everything here is original code against the public
vLLM/safetensors/PEFT-format APIs, not adapted from third_party/eggroll (GPL-3.0).

## Why the update is applied in place

The first version of this module ended each adapter block by tearing down the
vLLM engine, merging the ES delta into a fresh 15 GB HF checkpoint on disk, and
rebuilding the engine on it. That cost ~15 min per block and, on the third engine
build in one process, reliably hung right after `Using V2 Model Runner` -- vLLM V1
spawns an EngineCore subprocess per build, and repeated build/teardown in one
process is a fragile path.

Instead the engine is now built **once per run** and ES updates are pushed
straight into the live GPU weights via `LLM.collective_rpc`. This removes the hang,
the per-block reload, and the per-block disk write, and makes fresh-noise-every-step
(steps_per_adapter=1) free.

Two invariants this relies on:

1. **The update must use bit-identical noise to the adapters that produced the
   fitness.** Adapters are written from the parent process on CPU; the worker
   therefore also generates on CPU (`torch.Generator()` with no device) before
   moving to GPU. A CUDA generator with the same seed produces *different* values,
   which would silently corrupt every update.
2. **The update must be in-place** (`param.data.add_`), never a reassignment.
   vLLM captures CUDA graphs over fixed parameter addresses; writing through the
   existing storage keeps those graphs valid.

Prefix caching is disabled on the training engine: KV cached under old weights
would be stale after an update, and it buys nothing here anyway (vLLM keys the
prefix cache per LoRA adapter, so the shared prompt batch is not reused across
population members).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from es_capacity.posttrain.eggroll import TARGET_MODULES, scaled_pair_noise

_ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


def lora_int_id(pop_step: int, num_pairs: int, pair_idx: int, is_pos: bool) -> int:
    """Globally unique LoRA adapter id for the whole run (never reused across
    steps or resumes, since pop_step only increases) -- avoids relying on vLLM
    to detect an id being reused for a different adapter path."""
    return 1 + pop_step * (num_pairs * 2) + pair_idx * 2 + (0 if is_pos else 1)


def _adapter_config(rank: int, model_path: str) -> dict[str, Any]:
    return {
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


def write_pair_adapters(
    model_path: str | Path,
    out_dir: Path,
    *,
    pair_indices: list[int],
    num_layers: int,
    shapes: dict[str, tuple[int, int]],
    lora_r: int,
    sigma: float,
    base_seed: int,
    pop_step: int,
    num_pairs: int,
) -> dict[int, tuple[Path, Path]]:
    """Write a (+B) and a (-B) PEFT-format adapter dir per antithetic pair,
    using real ES noise (scaled_pair_noise). Returns {pair_idx: (pos, neg)}."""
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    acfg = _adapter_config(lora_r, str(model_path))
    result: dict[int, tuple[Path, Path]] = {}
    for pair_idx in pair_indices:
        pos_tensors: dict[str, torch.Tensor] = {}
        neg_tensors: dict[str, torch.Tensor] = {}
        for layer in range(num_layers):
            for mod, in_out_shape in shapes.items():
                a, b = scaled_pair_noise(
                    base_seed, num_pairs, pair_idx, num_layers, layer, pop_step,
                    lora_r, sigma, in_out_shape,
                )
                block = "self_attn" if mod in _ATTN_MODULES else "mlp"
                stem = f"base_model.model.model.layers.{layer}.{block}.{mod}"
                pos_tensors[f"{stem}.lora_A.weight"] = a.to(torch.bfloat16)
                pos_tensors[f"{stem}.lora_B.weight"] = b.to(torch.bfloat16)
                neg_tensors[f"{stem}.lora_A.weight"] = a.to(torch.bfloat16)
                neg_tensors[f"{stem}.lora_B.weight"] = (-b).to(torch.bfloat16)
        pos_dir = out_dir / f"pair_{pair_idx:05d}_pos"
        neg_dir = out_dir / f"pair_{pair_idx:05d}_neg"
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)
        save_file(pos_tensors, str(pos_dir / "adapter_model.safetensors"))
        save_file(neg_tensors, str(neg_dir / "adapter_model.safetensors"))
        (pos_dir / "adapter_config.json").write_text(json.dumps(acfg, indent=2))
        (neg_dir / "adapter_config.json").write_text(json.dumps(acfg, indent=2))
        result[pair_idx] = (pos_dir, neg_dir)
    return result


def fused_targets(model_cfg: dict[str, Any]) -> dict[str, tuple[str, int]]:
    """Map each HF target module to (vLLM parameter suffix, row offset).

    vLLM fuses q/k/v into one `qkv_proj` and gate/up into one `gate_up_proj`
    (vllm/model_executor/models/qwen2.py), so an ES delta computed per HF module
    lands on a *row slice* of the fused tensor. Verified for tensor_parallel_size=1;
    with TP>1 each shard would additionally need its own row range.
    """
    hidden = int(model_cfg["hidden_size"])
    inter = int(model_cfg["intermediate_size"])
    n_heads = int(model_cfg["num_attention_heads"])
    n_kv = int(model_cfg.get("num_key_value_heads", n_heads))
    head_dim = int(model_cfg.get("head_dim", hidden // n_heads))
    q_size = n_heads * head_dim
    kv_size = n_kv * head_dim
    return {
        "q_proj": ("self_attn.qkv_proj", 0),
        "k_proj": ("self_attn.qkv_proj", q_size),
        "v_proj": ("self_attn.qkv_proj", q_size + kv_size),
        "o_proj": ("self_attn.o_proj", 0),
        "gate_proj": ("mlp.gate_up_proj", 0),
        "up_proj": ("mlp.gate_up_proj", inter),
        "down_proj": ("mlp.down_proj", 0),
    }


class ESWorkerExtension:
    """Mixed into vLLM's Worker class at engine startup (see TrainEngine, which
    passes this class's *qualified name string* as `worker_extension_cls`).

    `collective_rpc` refuses to serialize an arbitrary function object by default
    (`TypeError: ... is not serializable`, `VLLM_ALLOW_INSECURE_SERIALIZATION`
    notwithstanding -- that flag falls back to pickle, which is the wrong fix for
    trusted, already-imported code). The supported path is to give the RPC a
    *method name string* to call on the worker, which is how vLLM's own LoRA/tuning
    integrations do it. Because this class is mixed into the real Worker's MRO,
    `self.get_model()` etc. are simply inherited, not re-implemented.
    """

    def list_param_names(self, pattern: str = "") -> list[str]:
        """List live parameter names, optionally filtered by substring. Exists
        because vLLM's internal naming can differ from the HF checkpoint's
        (fused qkv/gate_up, wrapper prefixes) -- use this rather than assuming."""
        return [n for n, _ in self.get_model().named_parameters() if pattern in n]

    def read_params(self, names: list[str]) -> dict[str, torch.Tensor]:
        """Fetch named parameters back to CPU. Used to verify apply_es_update
        against an independently computed expected delta, and as a general
        live-weight inspection hook (e.g. confirming an update actually moved
        the weights it was supposed to)."""
        params = dict(self.get_model().named_parameters())
        return {n: params[n].data.detach().float().cpu().clone() for n in names}

    def apply_es_update(self, spec: dict[str, Any]) -> dict[str, float]:
        """Applies  W += scale * sum_pairs fitness_diff_k * (B_k @ A_k)  to the
        live GPU weights, regenerating (A_k, B_k) from seeds rather than shipping
        tensors over RPC (the full dense delta would be ~15 GB; the scalars are a
        few KB).

        Batches all pairs into one GEMM per (layer, module) instead of one outer
        product per pair: stack A_all [P*r, in], B_all [out, P*r], scale B_all's
        columns by the per-pair diff, then (B_all * diff) @ A_all in one matmul.
        """
        import torch as _torch

        from es_capacity.posttrain.eggroll import scaled_pair_noise as _noise

        model = self.get_model()
        params = dict(model.named_parameters())

        shapes: dict[str, tuple[int, int]] = {k: tuple(v) for k, v in spec["shapes"].items()}
        targets: dict[str, tuple[str, int]] = {k: tuple(v) for k, v in spec["fused_targets"].items()}
        pairs: list[tuple[int, int, float]] = spec["pairs"]  # (pop_step, pair_idx, diff)
        num_layers = int(spec["num_layers"])
        num_pairs = int(spec["num_pairs"])
        lora_r = int(spec["lora_r"])
        sigma = float(spec["sigma"])
        base_seed = int(spec["base_seed"])
        scale = float(spec["scale"])

        if not pairs:
            return {"applied": 0.0, "max_abs_delta": 0.0}

        max_abs = 0.0
        n_applied = 0
        for layer in range(num_layers):
            for mod, in_out_shape in shapes.items():
                suffix, row_off = targets[mod]
                # `.base_layer` because TrainEngine always builds with enable_lora=True:
                # vLLM wraps every LoRA-target module in a BaseLayerWithLoRA, which holds
                # the real weight at `<module>.base_layer.weight` -- confirmed against a
                # live model's named_parameters(), not assumed from the HF checkpoint's
                # (unwrapped) naming.
                pname = f"model.layers.{layer}.{suffix}.base_layer.weight"
                param = params.get(pname)
                if param is None:
                    raise KeyError(f"vLLM parameter not found: {pname}")
                _, out_f = in_out_shape

                # CPU generation, to stay bit-identical with the written adapters.
                a_rows, b_cols, diffs = [], [], []
                for pop_step, pair_idx, diff in pairs:
                    a, b = _noise(
                        base_seed, num_pairs, pair_idx, num_layers, layer, pop_step,
                        lora_r, sigma, in_out_shape,
                    )
                    a_rows.append(a)  # [r, in]
                    b_cols.append(b)  # [out, r]
                    diffs.append(diff)
                a_all = _torch.cat(a_rows, dim=0).to(param.device, _torch.float32)  # [P*r, in]
                b_all = _torch.cat(b_cols, dim=1).to(param.device, _torch.float32)  # [out, P*r]
                d = _torch.tensor(diffs, device=param.device, dtype=_torch.float32)
                if lora_r != 1:
                    d = d.repeat_interleave(lora_r)
                delta = (b_all * d) @ a_all  # [out, in]

                target = param.data[row_off : row_off + out_f]
                if target.shape != delta.shape:
                    raise ValueError(
                        f"{pname}[{row_off}:{row_off+out_f}] has shape {tuple(target.shape)}, "
                        f"delta is {tuple(delta.shape)}"
                    )
                target.add_((scale * delta).to(target.dtype))  # in-place: keeps CUDA graphs valid
                max_abs = max(max_abs, float((scale * delta).abs().max()))
                n_applied += 1

        return {"applied": float(n_applied), "max_abs_delta": max_abs}


class TrainEngine:
    """One long-lived vLLM engine with Multi-LoRA enabled, whose base weights are
    mutated in place between ES steps. Built once per run -- see module docstring."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_model_len: int,
        max_num_seqs: int,
        max_loras: int,
        max_lora_rank: int,
        gpu_memory_utilization: float,
        dtype: str = "bfloat16",
    ):
        from vllm import LLM

        self.model_path = str(model_path)
        self.llm = LLM(
            model=self.model_path,
            dtype=dtype,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            max_num_seqs=int(max_num_seqs),
            enable_lora=True,
            max_loras=int(max_loras),
            max_lora_rank=int(max_lora_rank),
            enable_prefix_caching=False,  # would go stale under in-place updates
            trust_remote_code=True,
            seed=0,
            worker_extension_cls="es_capacity.posttrain.engine.ESWorkerExtension",
        )

    def generate_chunk(
        self,
        prompts: list[str],
        adapter_paths: list[Path],
        adapter_ids: list[int],
        *,
        temperature: float,
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        """prompts/adapter_paths/adapter_ids are pre-expanded 1:1 (member-major).
        Returns one dict per request with text, token count and finish reason --
        the token stats feed the per-step diagnostics that make a stalled or
        degenerate run visible without reading vLLM's log."""
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=1)
        lora_reqs = [
            LoRARequest(f"m{aid}", aid, str(apath)) for aid, apath in zip(adapter_ids, adapter_paths)
        ]
        outs = self.llm.generate(prompts, sp, lora_request=lora_reqs)
        return [
            {
                "text": o.outputs[0].text,
                "num_tokens": len(o.outputs[0].token_ids),
                "finish_reason": str(o.outputs[0].finish_reason),
            }
            for o in outs
        ]

    def apply_es_update(
        self,
        pairs: list[tuple[int, int, float]],
        *,
        model_cfg: dict[str, Any],
        shapes: dict[str, tuple[int, int]],
        num_layers: int,
        num_pairs: int,
        lora_r: int,
        sigma: float,
        base_seed: int,
        scale: float,
        timeout: float = 900.0,
    ) -> dict[str, float]:
        """Push one ES update into the live GPU weights. `timeout` bounds the RPC
        so a wedged worker surfaces as an exception instead of an overnight hang."""
        spec = {
            "shapes": {k: list(v) for k, v in shapes.items()},
            "fused_targets": {k: list(v) for k, v in fused_targets(model_cfg).items()},
            "pairs": [(int(p), int(i), float(d)) for p, i, d in pairs],
            "num_layers": num_layers,
            "num_pairs": num_pairs,
            "lora_r": lora_r,
            "sigma": sigma,
            "base_seed": base_seed,
            "scale": scale,
        }
        results = self.llm.collective_rpc(
            "apply_es_update", timeout=timeout, args=(spec,)
        )
        return results[0]

    def close(self) -> None:
        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def materialize_checkpoint(
    base_ckpt_dir: Path,
    dst_ckpt_dir: Path,
    *,
    records: list[tuple[int, int, float]],
    num_layers: int,
    shapes: dict[str, tuple[int, int]],
    lora_r: int,
    sigma: float,
    base_seed: int,
    num_pairs: int,
    scale: float,
) -> None:
    """Replay ES records onto the base model to produce an evaluable HF checkpoint.

    Because the noise is seed-derived, the whole training trajectory is recoverable
    from the scalar records -- so `records.jsonl` is the real checkpoint and this is
    just how it gets turned into something vLLM can load for evaluation. Streams one
    safetensors shard at a time; non-target tensors and non-weight files are copied
    through unchanged.
    """
    import re
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    weight_re = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(" + "|".join(TARGET_MODULES) + r")\.weight$"
    )

    base_ckpt_dir = Path(base_ckpt_dir)
    dst_ckpt_dir = Path(dst_ckpt_dir)
    dst_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Sharded checkpoints carry an index; small models ship a single
    # model.safetensors with no index at all (e.g. Qwen2.5-1.5B).
    index_path = base_ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    else:
        single = base_ckpt_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"No safetensors checkpoint found in {base_ckpt_dir}")
        with safe_open(str(single), framework="pt", device="cpu") as f:
            weight_map = {name: single.name for name in f.keys()}

    by_pair: dict[tuple[int, int], float] = {}
    for pop_step, pair_idx, diff in records:
        key = (int(pop_step), int(pair_idx))
        by_pair[key] = by_pair.get(key, 0.0) + float(diff)
    active = [(p, i, d) for (p, i), d in by_pair.items() if d != 0.0]

    shards: dict[str, list[str]] = {}
    for tensor_name, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(tensor_name)

    for shard_file, tensor_names in shards.items():
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(base_ckpt_dir / shard_file), framework="pt", device="cpu") as f:
            for name in tensor_names:
                tensor = f.get_tensor(name)
                m = weight_re.match(name)
                if m is None:
                    out_tensors[name] = tensor
                    continue
                layer = int(m.group(1))
                in_f, out_f = shapes[m.group(2)]
                delta = torch.zeros((out_f, in_f), dtype=torch.float32)
                for pop_step, pair_idx, total_diff in active:
                    a, b = scaled_pair_noise(
                        base_seed, num_pairs, pair_idx, num_layers, layer, pop_step,
                        lora_r, sigma, (in_f, out_f),
                    )
                    delta += total_diff * (b.float() @ a.float())
                out_tensors[name] = (tensor.float() + scale * delta).to(tensor.dtype)
        save_file(out_tensors, str(dst_ckpt_dir / shard_file), metadata={"format": "pt"})

    for item in base_ckpt_dir.iterdir():
        if item.name in shards or not item.is_file():
            continue
        shutil.copy2(item, dst_ckpt_dir / item.name)
