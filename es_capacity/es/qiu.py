"""Qiu et al. full-parameter ES for LLM fine-tuning (arXiv:2509.24372).

OpenAI-ES / NES-style update over full weights:
  - Gaussian noise retrieved from random seeds (memory-cheap)
  - layer-wise in-place perturb / restore
  - reward normalization across the population
  - θ ← θ + α (1/N) Σ R_n ε_n  (σ absorbed into α in their impl)

See Qiu Algorithms 1–2. Outline only — no distributed workers yet.
"""

from __future__ import annotations

from typing import Any

from es_capacity.config import ESConfig


class QiuESTrainer:
    """Full-rank parameter-space ES (Qiu et al.)."""

    def __init__(self, model: Any, cfg: ESConfig) -> None:
        self.model = model
        self.cfg = cfg

    def sample_population(self) -> list[Any]:
        """Sample N full-rank Gaussian noise seeds / tensors.

        # TODO: store seeds only; regenerate ε ~ N(0, I) per layer as needed.
        """
        raise NotImplementedError("QiuESTrainer.sample_population")

    def evaluate_fitness(self, member: Any, batch: list[dict[str, Any]]) -> float:
        """Perturb in place → forward / reward → restore.

        # TODO: layer-level apply_noise / restore_noise; population_fitness.
        """
        raise NotImplementedError("QiuESTrainer.evaluate_fitness")

    def normalize_rewards(self, fitnesses: list[float]) -> list[float]:
        """Normalize population rewards before the weighted update.

        # TODO: z-score / rank normalization as in Qiu impl.
        """
        raise NotImplementedError("QiuESTrainer.normalize_rewards")

    def aggregate_update(self, members: list[Any], fitnesses: list[float]) -> None:
        """NES-style mean update from normalized rewards and ε_n.

        # TODO: reconstruct noise from seeds; θ += α * mean_n(R_n * ε_n).
        """
        raise NotImplementedError("QiuESTrainer.aggregate_update")

    def step(self, batch: list[dict[str, Any]]) -> dict[str, float]:
        """One Qiu-ES iteration.

        Outline:
          members = self.sample_population()
          fitnesses = [self.evaluate_fitness(m, batch) for m in members]
          fitnesses = self.normalize_rewards(fitnesses)
          self.aggregate_update(members, fitnesses)
        """
        raise NotImplementedError("QiuESTrainer.step")
