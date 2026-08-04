"""Unbiased pass@k, histogram, coverage, and shard aggregation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def passk_curve(
    correct_counts: list[int] | np.ndarray,
    n: int,
    ks: list[int] | None = None,
) -> dict[int, float]:
    correct_counts = np.asarray(correct_counts, dtype=np.int64)
    if ks is None:
        ks = [2**i for i in range(int(np.floor(np.log2(max(n, 1)))) + 1) if 2**i <= n]
        if n not in ks:
            ks.append(n)
    out: dict[int, float] = {}
    for k in ks:
        if k > n or k < 1:
            continue
        vals = [estimate_pass_at_k(n, int(c), k) for c in correct_counts]
        out[int(k)] = float(np.mean(vals)) if vals else 0.0
    return out


def accuracy_histogram(
    correct_counts: list[int] | np.ndarray,
    n: int,
    bins: int = 10,
) -> dict[str, Any]:
    """Per-problem accuracy c/n histogram."""
    acc = np.asarray(correct_counts, dtype=np.float64) / max(n, 1)
    counts, edges = np.histogram(acc, bins=bins, range=(0.0, 1.0))
    return {
        "bins": bins,
        "n": n,
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "mean_acc": float(acc.mean()) if len(acc) else 0.0,
        "per_problem_acc": acc.tolist(),
    }


def solvable_mask(correct_counts: list[int] | np.ndarray, threshold: int = 1) -> list[bool]:
    return [int(c) >= threshold for c in correct_counts]


def coverage_contingency(
    base_mask: list[bool],
    other_mask: list[bool],
    *,
    base_label: str = "base",
    other_label: str = "other",
) -> dict[str, Any]:
    """2x2 solvable-set overlap. Keys: both, base_only, other_only, neither."""
    assert len(base_mask) == len(other_mask)
    both = base_only = other_only = neither = 0
    for b, o in zip(base_mask, other_mask):
        if b and o:
            both += 1
        elif b and not o:
            base_only += 1
        elif o and not b:
            other_only += 1
        else:
            neither += 1
    total = len(base_mask)
    return {
        "base_label": base_label,
        "other_label": other_label,
        "n_problems": total,
        "both": both,
        "base_only": base_only,
        "other_only": other_only,
        "neither": neither,
        "other_subset_of_base": other_only == 0,
        "frac_other_only": other_only / total if total else 0.0,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate_run(run_dir: Path, ks: list[int] | None = None) -> dict[str, Any]:
    """Sum correct counts across complete shards; refuse mismatched sampling params."""
    run_dir = Path(run_dir)
    shards_root = run_dir / "shards"
    if not shards_root.exists():
        raise FileNotFoundError(f"No shards under {run_dir}")

    sampling_ref: dict[str, Any] | None = None
    model_ref: str | None = None
    per_idx: dict[int, list[int]] = defaultdict(list)  # idx -> list of c per shard
    n_per_shard: int | None = None
    complete_shards: list[int] = []

    for sdir in sorted(shards_root.glob("shard_*")):
        man_path = sdir / "manifest.json"
        rec_path = sdir / "records.jsonl"
        if not man_path.exists() or not rec_path.exists():
            continue
        man = json.loads(man_path.read_text())
        if not man.get("complete"):
            continue
        samp = man.get("sampling")
        if sampling_ref is None:
            sampling_ref = samp
            model_ref = man.get("model_path")
            n_per_shard = int(man.get("shard_size", samp.get("shard_size", 0) if samp else 0))
        else:
            if samp != sampling_ref:
                raise ValueError(
                    f"Sampling mismatch in {sdir.name}: {samp} vs {sampling_ref}"
                )
            if man.get("model_path") != model_ref:
                raise ValueError(f"Model path mismatch in {sdir.name}")
        records = read_jsonl(rec_path)
        for r in records:
            per_idx[int(r["idx"])].append(int(r["c"]))
        complete_shards.append(int(man["shard_idx"]))

    if not complete_shards:
        raise RuntimeError(f"No complete shards in {run_dir}")

    # Each problem must have same number of shards
    n_shards = len(complete_shards)
    indices = sorted(per_idx)
    for idx in indices:
        if len(per_idx[idx]) != n_shards:
            raise ValueError(
                f"idx={idx} has {len(per_idx[idx])} shard counts; expected {n_shards}"
            )

    assert n_per_shard is not None
    n_total = n_per_shard * n_shards
    correct_counts = [int(sum(per_idx[idx])) for idx in indices]
    if ks is None:
        # Prefer manifest eval.ks if present
        run_man = {}
        if (run_dir / "manifest.json").exists():
            run_man = json.loads((run_dir / "manifest.json").read_text())
        ks = None  # passk_curve will auto-build

    curve = passk_curve(correct_counts, n_total, ks=ks)
    hist = accuracy_histogram(correct_counts, n_total)
    agg = {
        "run_dir": str(run_dir),
        "complete_shards": complete_shards,
        "n_shards": n_shards,
        "shard_size": n_per_shard,
        "n_total": n_total,
        "num_problems": len(indices),
        "indices": indices,
        "correct_counts": correct_counts,
        "sampling": sampling_ref,
        "model_path": model_ref,
        "passk": {str(k): v for k, v in curve.items()},
        "histogram": hist,
    }
    out_dir = run_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "correct_counts.json").write_text(
        json.dumps(
            {
                "indices": indices,
                "correct_counts": correct_counts,
                "n_total": n_total,
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / "metrics.json").write_text(json.dumps(agg, indent=2) + "\n")
    return agg


def load_correct_counts(run_dir: Path) -> tuple[list[int], list[int], int]:
    """Return (indices, correct_counts, n_total)."""
    path = Path(run_dir) / "aggregate" / "correct_counts.json"
    if not path.exists():
        aggregate_run(run_dir)
    data = json.loads(path.read_text())
    return data["indices"], data["correct_counts"], int(data["n_total"])


def truncate_to_common_n(
    arms: dict[str, Path],
) -> tuple[int, dict[str, tuple[list[int], list[int], int]]]:
    """Load arms and truncate conceptually to min n (requires shard_size alignment).

    For v1 we require identical shard_size across arms and take min(n_shards)*shard_size
    by re-aggregating only the first N shards if needed. Simpler approach: use min n_total
    and scale counts is WRONG. Instead re-read first k shards.
    """
    loaded = {name: load_correct_counts(path) for name, path in arms.items()}
    n_min = min(n for _, _, n in loaded.values())
    # If all equal, done
    if all(n == n_min for _, _, n in loaded.values()):
        return n_min, loaded

    # Re-aggregate truncating shards
    out: dict[str, tuple[list[int], list[int], int]] = {}
    for name, path in arms.items():
        truncated = _aggregate_first_n_samples(path, n_min)
        out[name] = truncated
    return n_min, out


def _aggregate_first_n_samples(run_dir: Path, n_target: int) -> tuple[list[int], list[int], int]:
    run_dir = Path(run_dir)
    shards = sorted((run_dir / "shards").glob("shard_*"))
    per_idx: dict[int, int] = defaultdict(int)
    n_acc = 0
    shard_size = None
    for sdir in shards:
        man = json.loads((sdir / "manifest.json").read_text())
        if not man.get("complete"):
            continue
        ss = int(man["shard_size"])
        if n_acc + ss > n_target:
            break
        for r in read_jsonl(sdir / "records.jsonl"):
            per_idx[int(r["idx"])] += int(r["c"])
        n_acc += ss
        shard_size = ss
    if n_acc < n_target:
        raise ValueError(f"{run_dir} only has n={n_acc}, need {n_target}")
    indices = sorted(per_idx)
    return indices, [per_idx[i] for i in indices], n_acc
