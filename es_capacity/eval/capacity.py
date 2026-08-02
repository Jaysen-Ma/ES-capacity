"""Compare base vs ES reasoning capacity via pass@k curves.

Yue et al.: RLVR often raises pass@1 but can narrow coverage at large k.
Here the same lens is applied to ES-trained models.
Also define sampling-efficiency gap Δ_SE ≈ pass@1(ES) − pass@k_max(base)
(or analogous base/ES pairing — finalize once experiments start).
"""

from __future__ import annotations

from typing import Any


def sampling_efficiency_gap(
    pass1_trained: float,
    pass_kmax_base: float,
) -> float:
    """Δ_SE between a trained model's pass@1 and base pass@k_max.

    # TODO: confirm exact definition / pairing used in reports.
    """
    raise NotImplementedError("sampling_efficiency_gap")


def compare_capacity(
    curves: dict[str, dict[int, float]],
    *,
    k_max: int = 256,
) -> dict[str, Any]:
    """Summarize base vs ES pass@k curves and Δ_SE.

    Expected `curves` keys: at least "base" and "es", each mapping k → pass@k.

    # TODO: crossing points, coverage gap at k_max, tables/plots hooks.
    """
    raise NotImplementedError(f"compare_capacity(k_max={k_max})")
