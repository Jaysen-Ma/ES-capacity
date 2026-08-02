"""Shared ES trainer protocol / loop skeleton.

Each step: sample population → evaluate fitness → aggregate parameter update.
Concrete methods (Qiu, EGGROLL) differ in how noise is sampled and applied.
"""

from __future__ import annotations

from typing import Any, Protocol

from es_capacity.config import ESConfig


class ESTrainer(Protocol):
    """Minimal interface for ES post-training loops."""

    cfg: ESConfig

    def sample_population(self) -> list[Any]:
        """Sample N parameter-space perturbations (or seeds that regenerate them)."""
        ...

    def evaluate_fitness(self, member: Any, batch: list[dict[str, Any]]) -> float:
        """Fitness of one population member on a training batch."""
        ...

    def aggregate_update(self, members: list[Any], fitnesses: list[float]) -> None:
        """Update mean parameters from weighted / normalized perturbations."""
        ...

    def step(self, batch: list[dict[str, Any]]) -> dict[str, float]:
        """One ES iteration: sample → evaluate → update.

        Outline:
          members = self.sample_population()
          fitnesses = [self.evaluate_fitness(m, batch) for m in members]
          self.aggregate_update(members, fitnesses)
        """
        ...


def run_training_loop(
    trainer: ESTrainer,
    batches: list[list[dict[str, Any]]],
    num_iterations: int,
) -> Any:
    """Run `num_iterations` of `trainer.step`.

    # TODO: checkpointing, logging, parallel workers.
    """
    raise NotImplementedError("run_training_loop")
