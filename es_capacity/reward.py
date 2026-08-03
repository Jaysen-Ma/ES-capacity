"""Verifiable rewards for ES fitness and pass@k checking.

Answer extraction and equality follow Yue et al. (limit-of-RLVR):
``extract_answer`` + ``math_equal`` / ``parse_ground_truth`` for AIME24.
"""

from __future__ import annotations

from typing import Any

from es_capacity.yue_math import extract_answer as yue_extract_answer
from es_capacity.yue_math import math_equal, parse_ground_truth


def extract_answer(completion: str, data_name: str = "aime24") -> str:
    """Extract the predicted answer string (Yue ``parser.extract_answer``)."""
    return yue_extract_answer(completion, data_name=data_name, use_last_number=True)


def gold_answer(example: dict[str, Any], data_name: str = "aime24") -> str:
    """Normalized gold answer via Yue ``parse_ground_truth``."""
    _gt_cot, gt = parse_ground_truth(example, data_name)
    return str(gt)


def verify(example: dict[str, Any], completion: str, data_name: str = "aime24") -> bool:
    """True iff Yue extraction + ``math_equal`` matches gold."""
    pred = extract_answer(completion, data_name=data_name)
    gold = gold_answer(example, data_name=data_name)
    if pred is None or pred == "":
        return False
    try:
        return bool(math_equal(pred, gold, timeout=True))
    except Exception:
        return str(pred).strip() == str(gold).strip()


def reward(example: dict[str, Any], completion: str) -> float:
    """Scalar fitness for one (prompt, completion) pair."""
    return 1.0 if verify(example, completion) else 0.0


def population_fitness(
    model: Any,
    batch: list[dict[str, Any]],
    *,
    num_samples: int = 1,
) -> float:
    """Aggregate fitness for one perturbed population member."""
    raise NotImplementedError("population_fitness")
