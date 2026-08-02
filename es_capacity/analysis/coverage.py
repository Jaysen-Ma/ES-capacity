"""Solvable-set overlap / reasoning-boundary narrowing checks.

At a fixed large k, which problems does the base solve that ES does not
(and vice versa)? Used to test whether ES expands or shrinks coverage.
"""

from __future__ import annotations

from typing import Any


def solvable_mask(
    per_problem_correct_counts: list[int],
    *,
    n_samples: int,
    k: int,
) -> list[bool]:
    """Mark problems as solvable under pass@k given sample counts.

    # TODO: threshold via estimate_pass_at_k or c > 0 for large-n probes.
    """
    raise NotImplementedError("solvable_mask")


def solvable_set_overlap(
    base_mask: list[bool],
    es_mask: list[bool],
) -> dict[str, Any]:
    """Overlap statistics between base and ES solvable sets.

    # TODO: |base ∩ es|, |base \\ es|, |es \\ base|, Jaccard, etc.
    """
    raise NotImplementedError("solvable_set_overlap")
