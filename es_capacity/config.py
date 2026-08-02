"""Experiment configuration placeholders.

Holds knobs for base model, ES method (qiu | eggroll), and pass@k evaluation
without wiring real training or hardware details yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ESMethod = Literal["qiu", "eggroll"]


@dataclass
class ESConfig:
    """Shared ES hyperparameters (algorithm outline only)."""

    method: ESMethod = "qiu"
    population_size: int = 30
    noise_scale: float = 0.001  # σ
    learning_rate: float = 5e-4  # α
    num_iterations: int = 0
    # EGGROLL-only: rank of per-member low-rank perturbation A B^T
    rank: int = 1


@dataclass
class EvalConfig:
    """pass@k evaluation settings (Yue et al.)."""

    ks: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256])
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 16_384
    # Proxy upper bound for Δ_SE vs base pass@k_max
    k_max: int = 256


@dataclass
class ExperimentConfig:
    """Top-level experiment config for capacity comparison."""

    base_model: str = ""
    dataset: str = ""
    es: ESConfig = field(default_factory=ESConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
