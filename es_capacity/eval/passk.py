"""pass@k estimation for reasoning-capacity boundaries (Yue et al.).

A problem counts as solved if any of k samples passes the verifier.
Use the unbiased estimator over n ≥ k samples with c correct.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from es_capacity.reward import verify


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator for one problem.

    Calculates ``1 - C(n - c, k) / C(n, k)`` via the stable product form
    used by Chen et al. / Yue et al.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n < 0 or c < 0:
        raise ValueError(f"n and c must be non-negative, got n={n}, c={c}")
    if c > n:
        raise ValueError(f"c cannot exceed n, got n={n}, c={c}")
    if k > n:
        raise ValueError(f"k cannot exceed n, got n={n}, k={k}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def passk_from_correct_counts(
    num_correct: Sequence[int],
    n_samples: int,
    ks: Sequence[int],
) -> dict[int, float]:
    """Average unbiased pass@k over problems given per-problem correct counts."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    for k in ks:
        if k > n_samples:
            raise ValueError(f"k={k} exceeds n_samples={n_samples}")
    out: dict[int, float] = {}
    for k in ks:
        vals = [estimate_pass_at_k(n_samples, int(c), int(k)) for c in num_correct]
        out[int(k)] = float(np.mean(vals)) if vals else 0.0
    return out


def evaluate_passk(
    engine: Any,
    dataset: list[dict[str, Any]],
    ks: list[int],
    *,
    n_samples: int | None = None,
    gen_kwargs: dict[str, Any] | None = None,
    prompts: list[str] | None = None,
) -> dict[int, float]:
    """Average pass@k over ``dataset`` for each k in ``ks``.

    Samples ``n_samples`` (default ``max(ks)``) completions per problem.
    Prefer the CLI script for resume/checkpointing on long AIME runs.
    """
    from es_capacity.model import generate

    if not ks:
        raise ValueError("ks must be non-empty")
    n = n_samples if n_samples is not None else max(ks)
    if prompts is None:
        raise ValueError("prompts must be provided (build via es_capacity.data.build_prompt)")
    if len(prompts) != len(dataset):
        raise ValueError("prompts and dataset length mismatch")

    kwargs = dict(gen_kwargs or {})
    completions_per_problem = generate(engine, prompts, n=n, **kwargs)

    num_correct: list[int] = []
    for example, comps in zip(dataset, completions_per_problem):
        c = sum(1 for text in comps if verify(example, text))
        num_correct.append(c)
    return passk_from_correct_counts(num_correct, n, ks)
