"""Qiu full-rank ES Perturber (v2 drop-in).

Mirrors es-at-scale worker_extension: seed-based torch.randn added in-place,
restored after eval, then θ += (α/N) Σ F̃ᵢ εᵢ with z-score fitness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QiuConfig:
    population_size: int = 16
    sigma: float = 0.001
    alpha: float | None = None  # default sigma/2
    base_seed: int = 0

    def __post_init__(self) -> None:
        if self.alpha is None:
            self.alpha = self.sigma / 2


def z_score(rewards: list[float]) -> list[float]:
    import numpy as np

    r = np.asarray(rewards, dtype=np.float64)
    return ((r - r.mean()) / (r.std() + 1e-8)).tolist()


def make_noise_like(param: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=param.device)
    gen.manual_seed(int(seed) % (2**31 - 1))
    return torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=gen)


class QiuPerturber:
    def __init__(self, cfg: QiuConfig | None = None):
        self.cfg = cfg or QiuConfig()
        self._engine = None

    def prepare_population(self, step: int) -> list[dict[str, Any]]:
        # Deterministic seeds per (step, member) — same idea as es-at-scale loop_rng
        g = torch.Generator()
        g.manual_seed(self.cfg.base_seed + step)
        seeds = torch.randint(0, 2**30, (self.cfg.population_size,), generator=g).tolist()
        return [{"seed": int(s), "step": step} for s in seeds]

    def perturb_param_(self, param: torch.Tensor, seed: int, *, negate: bool = False) -> None:
        noise = make_noise_like(param, seed)
        sign = -1.0 if negate else 1.0
        param.data.add_(sign * self.cfg.sigma * noise)

    def restore_param_(self, param: torch.Tensor, seed: int) -> None:
        self.perturb_param_(param, seed, negate=True)

    def accumulate_update_(
        self,
        param: torch.Tensor,
        seeds: list[int],
        coeffs: list[float],
    ) -> None:
        """θ += (α/N) Σ coeff_i * ε_i  (noise regenerated from seed)."""
        acc = torch.zeros_like(param.data, dtype=torch.float32)
        for seed, coeff in zip(seeds, coeffs):
            noise = make_noise_like(param, seed).to(torch.float32)
            acc.add_(noise * float(coeff))
        scale = float(self.cfg.alpha) / float(self.cfg.population_size)
        param.data.add_((acc * scale).to(param.dtype))

    def evaluate(self, members: list[Any], prompts: list[str], **kwargs) -> list[float]:
        raise NotImplementedError("Wire via single-engine ESLoop (v2)")

    def update(self, members: list[Any], fitnesses: list[float], step: int) -> None:
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError
