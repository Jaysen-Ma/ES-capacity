"""Compare reasoning capacity via pass@k curves.

Yue et al.: RLVR often raises pass@1 but can narrow coverage at large k.
Here the same lens is applied to base vs instruct / ES-trained models.
"""

from __future__ import annotations

from typing import Any


def sampling_efficiency_gap(
    pass1_trained: float,
    pass_kmax_base: float,
) -> float:
    """Δ_SE between a trained model's pass@1 and base pass@k_max."""
    return float(pass1_trained - pass_kmax_base)


def compare_capacity(
    curves: dict[str, dict[int, float]],
    *,
    k_max: int | None = None,
) -> dict[str, Any]:
    """Summarize pass@k curves side-by-side.

    ``curves`` maps a label (e.g. model slug) to ``{k: pass@k}``.
    """
    if not curves:
        raise ValueError("curves must be non-empty")

    all_ks: set[int] = set()
    for curve in curves.values():
        all_ks.update(int(k) for k in curve.keys())
    ks_sorted = sorted(all_ks)
    if k_max is None and ks_sorted:
        k_max = ks_sorted[-1]

    table: dict[str, dict[str, float]] = {}
    for name, curve in curves.items():
        table[name] = {f"pass@{k}": float(curve[k]) for k in ks_sorted if k in curve}

    summary: dict[str, Any] = {
        "ks": ks_sorted,
        "k_max": k_max,
        "curves": {name: {int(k): float(v) for k, v in curve.items()} for name, curve in curves.items()},
        "table": table,
    }

    names = list(curves.keys())
    if k_max is not None and len(names) >= 2 and 1 in curves[names[0]] and k_max in curves[names[0]]:
        # Optional Δ_SE when a "trained" vs "base" pairing is clear — leave unset
        # for generic multi-model comparisons (e.g. Base vs Instruct).
        summary["delta_se"] = None

    return summary


def format_capacity_table(summary: dict[str, Any]) -> str:
    """Pretty-print a compare_capacity summary as a text table."""
    ks: list[int] = list(summary["ks"])
    curves: dict[str, dict[int, float]] = summary["curves"]
    headers = ["model"] + [f"pass@{k}" for k in ks]
    rows: list[list[str]] = [headers]
    for name, curve in curves.items():
        row = [name] + [f"{curve.get(k, float('nan')) * 100:.1f}%" for k in ks]
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    lines = []
    for r in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)))
    return "\n".join(lines)
