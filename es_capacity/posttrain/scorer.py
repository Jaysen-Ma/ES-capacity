"""Train/eval reward via the same Yue grader used at evaluation."""

from __future__ import annotations

from es_capacity.grade import extract_answer, math_equal


class YueMathScorer:
    def __init__(self, data_name: str = "minerva_math"):
        self.data_name = data_name

    def score(self, completion: str, gold: str) -> float:
        pred = extract_answer(completion, self.data_name)
        try:
            return 1.0 if math_equal(pred, gold) else 0.0
        except Exception:
            return 0.0

    def score_batch(self, completions: list[str], golds: list[str]) -> list[float]:
        return [self.score(c, g) for c, g in zip(completions, golds)]
