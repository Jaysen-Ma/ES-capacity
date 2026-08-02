"""Verifiable rewards for ES fitness and pass@k checking.

Binary outcome reward R(x, y) ∈ {0, 1}, as in RLVR / Yue et al.:
correct final answer (math) or passing unit tests (code).
"""

from __future__ import annotations

from typing import Any


def verify(example: dict[str, Any], completion: str) -> bool:
    """Return True iff completion passes the task verifier.

    # TODO: math answer extraction / exact match; code unit tests.
    """
    raise NotImplementedError("verify")


def reward(example: dict[str, Any], completion: str) -> float:
    """Scalar fitness for one (prompt, completion) pair.

    Default shape: 1.0 if verify(...) else 0.0.
    # TODO: optional format reward; batch / population aggregation.
    """
    raise NotImplementedError("reward")


def population_fitness(
    model: Any,
    batch: list[dict[str, Any]],
    *,
    num_samples: int = 1,
) -> float:
    """Aggregate fitness for one perturbed population member.

    # TODO: sample responses under current weights; mean reward over batch.
    """
    raise NotImplementedError("population_fitness")
