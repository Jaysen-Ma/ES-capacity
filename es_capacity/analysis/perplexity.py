"""Probes for whether ES reasoning paths already live in the base model.

Yue et al. argue RLVR paths are typically already in the base sampling
distribution (perplexity / likelihood under the base). Stub for the same
check on ES-generated correct chains of thought.
"""

from __future__ import annotations

from typing import Any


def sequence_perplexity(model: Any, prompt: str, completion: str) -> float:
    """Perplexity (or NLL) of `completion` under `model` given `prompt`.

    # TODO: teacher-forced log-likelihood under base / ES models.
    """
    raise NotImplementedError("sequence_perplexity")


def path_in_base_distribution(
    base_model: Any,
    prompts: list[str],
    es_completions: list[str],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Summarize whether ES correct paths look in-distribution for the base.

    # TODO: compare base perplexity of ES vs base correct CoTs; set threshold.
    """
    raise NotImplementedError("path_in_base_distribution")
