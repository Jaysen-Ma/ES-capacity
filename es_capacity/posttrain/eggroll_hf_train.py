"""Minimal single-GPU EGGROLL trainer (HF generate, in-place low-rank updates).

Used to validate rising reward on 1.5B before investing in 7B / Multi-LoRA Ray training.
Applies antithetic low-rank noise directly to Linear weights (no Multi-LoRA required).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from es_capacity.datasets import build_prompt, load_minerva_math
from es_capacity.posttrain.eggroll import EggrollConfig, EggrollPerturber, get_rng_noise
from es_capacity.posttrain.scorer import YueMathScorer


def _iter_linear_params(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(
            t in name for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        ):
            yield name, module.weight


@torch.no_grad()
def apply_lowrank_noise_(
    model,
    *,
    pair_idx: int,
    sign: int,
    step: int,
    cfg: EggrollConfig,
    num_layers: int,
) -> None:
    for layer_idx, (name, w) in enumerate(_iter_linear_params(model)):
        out_f, in_f = w.shape
        r = cfg.lora_r
        shapes = ((r, in_f), (out_f, r))
        noise_a, noise_b = get_rng_noise(
            cfg.base_seed, cfg.population_size // 2, pair_idx, num_layers, layer_idx, step, shapes
        )
        noise_a = noise_a.to(device=w.device, dtype=w.dtype) * math.sqrt(cfg.sigma)
        noise_b = noise_b.to(device=w.device, dtype=w.dtype) * math.sqrt(cfg.sigma / cfg.lora_r)
        delta = sign * (noise_b @ noise_a)
        w.add_(delta)


@torch.no_grad()
def apply_es_update_(
    model,
    *,
    diffs: list[float],
    step: int,
    cfg: EggrollConfig,
    num_layers: int,
) -> None:
    n = cfg.population_size
    for layer_idx, (name, w) in enumerate(_iter_linear_params(model)):
        out_f, in_f = w.shape
        r = cfg.lora_r
        shapes = ((r, in_f), (out_f, r))
        acc = torch.zeros_like(w, dtype=torch.float32)
        for pair_idx, diff in enumerate(diffs):
            noise_a, noise_b = get_rng_noise(
                cfg.base_seed, cfg.population_size // 2, pair_idx, num_layers, layer_idx, step, shapes
            )
            noise_a = noise_a.to(device=w.device, dtype=torch.float32) * math.sqrt(cfg.sigma)
            noise_b = noise_b.to(device=w.device, dtype=torch.float32) * math.sqrt(cfg.sigma / cfg.lora_r)
            acc.add_(float(diff) * (noise_b @ noise_a))
        scale = cfg.learning_rate / (n * cfg.sigma + 1e-8)
        w.add_((acc * scale).to(w.dtype))


def generate_texts(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    texts = []
    for i, seq in enumerate(out):
        prompt_len = inputs["input_ids"][i].shape[0]
        texts.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
    return texts


def _easy_problems(n: int = 4) -> list[dict]:
    """Synthetic easy boxed-answer problems for loop validation."""
    bank = [
        {"idx": 0, "problem": "What is 2 + 2? Write the answer in \\boxed{}.", "solution": "\\boxed{4}"},
        {"idx": 1, "problem": "What is 3 + 5? Write the answer in \\boxed{}.", "solution": "\\boxed{8}"},
        {"idx": 2, "problem": "What is 10 - 4? Write the answer in \\boxed{}.", "solution": "\\boxed{6}"},
        {"idx": 3, "problem": "What is 7 + 1? Write the answer in \\boxed{}.", "solution": "\\boxed{8}"},
        {"idx": 4, "problem": "What is 9 - 3? Write the answer in \\boxed{}.", "solution": "\\boxed{6}"},
        {"idx": 5, "problem": "What is 5 + 5? Write the answer in \\boxed{}.", "solution": "\\boxed{10}"},
    ]
    return bank[:n]


def train_eggroll_hf(
    model_path: str,
    *,
    out_dir: str | Path,
    population_size: int = 8,
    num_iterations: int = 20,
    num_problems: int = 4,
    max_new_tokens: int = 256,
    lora_r: int = 1,
    sigma: float = 0.001,
    learning_rate: float = 0.0002,
    easy_synthetic: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = EggrollConfig(
        population_size=population_size,
        lora_r=lora_r,
        sigma=sigma,
        learning_rate=learning_rate,
    )
    pert = EggrollPerturber(cfg)
    scorer = YueMathScorer()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    num_layers = sum(1 for _ in _iter_linear_params(model))

    if easy_synthetic:
        problems = _easy_problems(num_problems)
        golds = ["4", "8", "6", "8", "6", "10"][:num_problems]
    else:
        problems = load_minerva_math(num_problems=num_problems)
        from es_capacity.datasets import gold_from_example

        golds = [gold_from_example(p) for p in problems]
    prompts = [build_prompt(p) for p in problems]

    history = []
    t0 = time.time()
    for step in range(num_iterations):
        members = pert.prepare_population(step)
        fitnesses = []
        for m in members:
            apply_lowrank_noise_(
                model,
                pair_idx=m["pair_idx"],
                sign=m["sign"],
                step=step,
                cfg=cfg,
                num_layers=num_layers,
            )
            texts = generate_texts(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            rewards = [scorer.score(t, g) for t, g in zip(texts, golds)]
            fitnesses.append(float(sum(rewards) / max(len(rewards), 1)))
            # restore
            apply_lowrank_noise_(
                model,
                pair_idx=m["pair_idx"],
                sign=-m["sign"],
                step=step,
                cfg=cfg,
                num_layers=num_layers,
            )
        normed = pert.normalize_fitnesses(fitnesses)
        diffs = pert.antithetic_diffs(normed)
        apply_es_update_(model, diffs=diffs, step=step, cfg=cfg, num_layers=num_layers)
        mean_r = float(sum(fitnesses) / len(fitnesses))
        row = {"step": step, "mean_reward": mean_r, "fitnesses": fitnesses}
        history.append(row)
        with (out_dir / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[eggroll-hf] step={step} mean_reward={mean_r:.4f}")

    # Save
    ckpt = out_dir / "checkpoint"
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    rising = False
    if len(history) >= 6:
        early = sum(h["mean_reward"] for h in history[:5]) / 5
        late = sum(h["mean_reward"] for h in history[-5:]) / 5
        rising = late > early + 1e-4
    summary = {
        "mean_reward_first": history[0]["mean_reward"] if history else None,
        "mean_reward_last": history[-1]["mean_reward"] if history else None,
        "reward_rising": rising,
        "wall_sec": time.time() - t0,
        "checkpoint": str(ckpt),
        "num_iterations": num_iterations,
        "population_size": population_size,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Model path; defaults to config models.<--model-key>")
    p.add_argument("--model-key", default="base_1p5b")
    p.add_argument("--machine", default=None, help="Default: $ES_CAPACITY_MACHINE or 'example'")
    p.add_argument("--out-dir", default="runs/train/eggroll_1p5b_hf_validate")
    p.add_argument("--population-size", type=int, default=8)
    p.add_argument("--num-iterations", type=int, default=15)
    p.add_argument("--num-problems", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--easy-synthetic", action="store_true")
    args = p.parse_args()
    model_path = args.model
    if model_path is None:
        from es_capacity.config import load_config

        model_path = str(load_config(machine=args.machine).model(args.model_key).path)
    s = train_eggroll_hf(
        model_path,
        out_dir=args.out_dir,
        population_size=args.population_size,
        num_iterations=args.num_iterations,
        num_problems=args.num_problems,
        max_new_tokens=args.max_new_tokens,
        easy_synthetic=args.easy_synthetic,
    )
    print(json.dumps(s, indent=2))
