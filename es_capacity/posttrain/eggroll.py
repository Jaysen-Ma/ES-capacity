"""EGGROLL low-rank Perturber (v1).

Implements antithetic seed-based LoRA noise following eggroll-vllm's
`get_rng_noise` / Multi-LoRA pattern. Full Ray multi-engine training is
orchestrated by ESLoop; this class owns the algorithm math.

Multi-LoRA on sm_121 must be probed before calling evaluate().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class EggrollConfig:
    population_size: int = 64  # even
    lora_r: int = 1
    sigma: float = 0.001
    learning_rate: float = 0.0002
    normalize_with_std: bool = True
    base_seed: int = 0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def get_rng_noise(
    base_seed: int,
    num_pop_pairs: int,
    pop_pair_idx: int,
    num_layers: int,
    layer_idx: int,
    step: int,
    shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seed-regenerated (A, B) noise — mirrors eggroll-vllm get_rng_noise."""
    noise_id = base_seed + (num_pop_pairs * num_layers * step) + (pop_pair_idx * num_layers) + layer_idx
    gen = torch.Generator()
    gen.manual_seed(int(noise_id) % (2**63 - 1))
    shape_a, shape_b = shapes
    noise_a = torch.normal(mean=0.0, std=1.0, size=shape_a, generator=gen)
    noise_b = torch.normal(mean=0.0, std=1.0, size=shape_b, generator=gen)
    return noise_a, noise_b


class EggrollPerturber:
    """Low-rank antithetic ES via LoRA-factorized noise.

    evaluate() requires a vLLM engine with enable_lora=True and Multi-LoRA support.
    Until the engine is wired, prepare_population / compute_update expose the math.
    """

    def __init__(self, cfg: EggrollConfig | None = None):
        self.cfg = cfg or EggrollConfig()
        if self.cfg.population_size % 2 != 0:
            raise ValueError("population_size must be even (antithetic pairs)")
        self._engine = None
        self._peft_shapes: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}

    @property
    def num_pairs(self) -> int:
        return self.cfg.population_size // 2

    def prepare_population(self, step: int) -> list[dict[str, Any]]:
        """Return metadata for each population member (pair index + sign)."""
        members = []
        for i in range(self.cfg.population_size):
            members.append(
                {
                    "pop_idx": i,
                    "pair_idx": i // 2,
                    "sign": 1 if i % 2 == 0 else -1,
                    "step": step,
                }
            )
        return members

    def lora_update_term(
        self,
        weight_shape: tuple[int, int],
        pair_idx: int,
        layer_idx: int,
        step: int,
        fitness_diff: float,
        num_layers: int,
    ) -> torch.Tensor:
        """Weighted low-rank update contribution for one antithetic pair on one layer.

        ΔW += fitness_diff * (noise_b @ noise_a) with σ scaling as in eggroll-vllm.
        """
        out_f, in_f = weight_shape
        r = self.cfg.lora_r
        shapes = ((r, in_f), (out_f, r))  # A: r×in, B: out×r
        noise_a, noise_b = get_rng_noise(
            self.cfg.base_seed,
            self.num_pairs,
            pair_idx,
            num_layers,
            layer_idx,
            step,
            shapes,
        )
        noise_a = noise_a * math.sqrt(self.cfg.sigma)
        noise_b = noise_b * math.sqrt(self.cfg.sigma / self.cfg.lora_r)
        if r == 1:
            # out×1 @ 1×in
            delta = (noise_b * fitness_diff) @ noise_a
        else:
            delta = fitness_diff * (noise_b @ noise_a)
        return delta

    def normalize_fitnesses(self, fitnesses: list[float], *, prompt_grouped: bool = False) -> list[float]:
        """Per-population mean-center (+ optional std), returning length=population."""
        import numpy as np

        f = np.asarray(fitnesses, dtype=np.float64)
        f = f - f.mean()
        if self.cfg.normalize_with_std:
            std = f.std()
            if std > 1e-8:
                f = f / std
        return f.tolist()

    def antithetic_diffs(self, normalized: list[float]) -> list[float]:
        """fitness_diff = f[2k] - f[2k+1] for each pair."""
        diffs = []
        for k in range(self.num_pairs):
            diffs.append(normalized[2 * k] - normalized[2 * k + 1])
        return diffs

    def evaluate(self, members: list[Any], prompts: list[str], **kwargs) -> list[float]:
        if self._engine is None:
            raise RuntimeError(
                "EggrollPerturber.evaluate requires a Multi-LoRA vLLM engine. "
                "Call attach_engine() after probe_multilora() succeeds."
            )
        raise NotImplementedError("Wire via ESLoop once Multi-LoRA probe passes")

    def attach_engine(self, engine: Any) -> None:
        self._engine = engine

    def update(self, members: list[Any], fitnesses: list[float], step: int) -> None:
        raise NotImplementedError("Applied inside engine WorkerExtension / ESLoop")

    def save_checkpoint(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._engine is None:
            raise RuntimeError("No engine attached")
        raise NotImplementedError
