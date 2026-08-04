"""Thin wrapper over third_party/yue_math grading.

Uses extract_answer + math_equal with a process pool sized from config.
Does not call evaluate.py's ProcessPool (that path is patched separately).
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

from es_capacity.config import REPO_ROOT

_YUE = REPO_ROOT / "third_party" / "yue_math"


def _ensure_yue_path() -> None:
    if str(_YUE) not in sys.path:
        sys.path.insert(0, str(_YUE))


def extract_answer(completion: str, data_name: str = "minerva_math") -> str:
    _ensure_yue_path()
    from parser import extract_answer as _extract  # type: ignore

    return _extract(completion, data_name)


def math_equal(prediction: str, reference: str, timeout: bool = False) -> bool:
    _ensure_yue_path()
    from grader import math_equal as _eq  # type: ignore

    return bool(_eq(prediction, reference, timeout=timeout))


def _grade_one(args: tuple[str, str, str]) -> tuple[str, bool]:
    """Worker: (completion, gold, data_name) -> (pred, correct)."""
    completion, gold, data_name = args
    _ensure_yue_path()
    from parser import extract_answer as _extract  # type: ignore
    from grader import math_equal as _eq  # type: ignore

    pred = _extract(completion, data_name)
    try:
        # timeout=True routes through grader.call_with_timeout, which runs the
        # symbolic-equality check in a child process and terminates it if it
        # doesn't return in time. timeout=False calls sympy in-process with no
        # bound and can hang forever on pathological expressions; our outer
        # ProcessPoolExecutor.result(timeout=...) only stops *us* from waiting,
        # it does not kill the stuck worker, so that combination can deadlock
        # ProcessPoolExecutor.shutdown() forever (observed 2026-08-04).
        ok = bool(_eq(pred, gold, timeout=True))
    except Exception:
        ok = False
    return pred, ok


def grade_completions(
    completions: list[str],
    gold: str,
    *,
    data_name: str = "minerva_math",
    num_workers: int = 16,
    timeout_sec: float = 3.0,
) -> dict[str, Any]:
    """Grade n completions for one problem. Returns preds, scores, c."""
    if not completions:
        return {"preds": [], "scores": [], "c": 0}

    # Serial for tiny batches to avoid process spawn overhead in smoke tests
    if len(completions) <= 2 or num_workers <= 1:
        preds, scores = [], []
        for c in completions:
            pred, ok = _grade_one((c, gold, data_name))
            preds.append(pred)
            scores.append(ok)
        return {"preds": preds, "scores": scores, "c": int(sum(scores))}

    args = [(c, gold, data_name) for c in completions]
    preds: list[str] = []
    scores: list[bool] = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_grade_one, a) for a in args]
        for fut in futures:
            try:
                pred, ok = fut.result(timeout=timeout_sec)
            except (FuturesTimeout, Exception):
                pred, ok = "", False
            preds.append(pred)
            scores.append(bool(ok))
    return {"preds": preds, "scores": scores, "c": int(sum(scores))}


def grade_batch(
    problems: list[dict[str, Any]],
    completions_per_problem: list[list[str]],
    *,
    data_name: str = "minerva_math",
    golds: list[str] | None = None,
    num_workers: int = 16,
    timeout_sec: float = 3.0,
) -> list[dict[str, Any]]:
    """Grade a list of problems. Each entry gets preds/scores/c."""
    from es_capacity.datasets import gold_from_example

    if golds is None:
        golds = [gold_from_example(p, data_name) for p in problems]

    # Flatten for one pool
    flat_args: list[tuple[str, str, str]] = []
    lengths: list[int] = []
    for comps, gold in zip(completions_per_problem, golds):
        lengths.append(len(comps))
        for c in comps:
            flat_args.append((c, gold, data_name))

    results: list[tuple[str, bool]] = []
    if not flat_args:
        return [{"preds": [], "scores": [], "c": 0, "gold": g} for g in golds]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_grade_one, a) for a in flat_args]
        for fut in futures:
            try:
                results.append(fut.result(timeout=timeout_sec))
            except (FuturesTimeout, Exception):
                results.append(("", False))

    out: list[dict[str, Any]] = []
    offset = 0
    for n, gold in zip(lengths, golds):
        chunk = results[offset : offset + n]
        offset += n
        preds = [p for p, _ in chunk]
        scores = [s for _, s in chunk]
        out.append({"preds": preds, "scores": scores, "c": int(sum(scores)), "gold": gold})
    return out
