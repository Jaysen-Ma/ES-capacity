"""Sarkar et al. EGGROLL: low-rank ES at hyperscale (arXiv:2511.16652).

Per population member, sample low-rank factors and form
  E = (1 / √r) A Bᵀ
instead of a full-rank matrix noise. Reconstruct from counter-based seeds;
batch fitness over the population; update the mean with the Gaussian
EGGROLL score-function approximation.

Overall update rank is min(N r, m, n) — not restricted to rank-r.
Outline only — no batched LoRA-style kernels yet.
"""

from __future__ import annotations

from typing import Any

from es_capacity.config import ESConfig


class EGGROLLTrainer:
    """Low-rank perturbation ES (EGGROLL / Sarkar et al.)."""

    def __init__(self, model: Any, cfg: ESConfig) -> None:
        self.model = model
        self.cfg = cfg

    def sample_population(self) -> list[Any]:
        """Sample N low-rank factor pairs (A_i, B_i) or regenerating seeds.

        # TODO: A ∈ R^{m×r}, B ∈ R^{n×r}; E = (1/√r) A Bᵀ per layer.
        """
        raise NotImplementedError("EGGROLLTrainer.sample_population")

    def evaluate_fitness(self, member: Any, batch: list[dict[str, Any]]) -> float:
        """Evaluate f(μ + σ E) for one low-rank member.

        # TODO: efficient batched forward with shared base activations.
        """
        raise NotImplementedError("EGGROLLTrainer.evaluate_fitness")

    def reconstruct_perturbation(self, seed: Any) -> Any:
        """Rebuild E_j from worker seed without materializing all noise in RAM.

        # TODO: counter-based RNG reconstruction across workers.
        """
        raise NotImplementedError("EGGROLLTrainer.reconstruct_perturbation")

    def aggregate_update(self, members: list[Any], fitnesses: list[float]) -> None:
        """EGGROLL mean update from population fitnesses and low-rank E_i.

        # TODO: μ ← μ + α (1/N) Σ f_i E_i  (constants absorbed into α).
        """
        raise NotImplementedError("EGGROLLTrainer.aggregate_update")

    def step(self, batch: list[dict[str, Any]]) -> dict[str, float]:
        """One EGGROLL iteration.

        Outline:
          members = self.sample_population()
          fitnesses = [self.evaluate_fitness(m, batch) for m in members]
          self.aggregate_update(members, fitnesses)
        """
        raise NotImplementedError("EGGROLLTrainer.step")
