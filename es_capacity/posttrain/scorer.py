"""Train/eval reward via the same Yue grader used at evaluation."""

from __future__ import annotations

from es_capacity.grade import extract_answer, grade_flat, math_equal


class YueMathScorer:
    """Binary correctness reward.

    Every symbolic-equality call goes through the grader's subprocess timeout
    path. Without it a single pathological expression spins forever in-process
    and wedges the whole training loop — this happened twice on the eval side
    (see third_party/yue_math/patches/0002-*.patch), and an ES step grades
    population x prompt_batch completions, so the exposure is far larger here.
    """

    def __init__(
        self,
        data_name: str = "minerva_math",
        *,
        num_workers: int = 16,
        timeout_sec: float = 3.0,
    ):
        self.data_name = data_name
        self.num_workers = num_workers
        self.timeout_sec = timeout_sec

    def score(self, completion: str, gold: str) -> float:
        pred = extract_answer(completion, self.data_name)
        try:
            return 1.0 if math_equal(pred, gold, timeout=True) else 0.0
        except Exception:
            return 0.0

    def score_batch(self, completions: list[str], golds: list[str]) -> list[float]:
        scores = grade_flat(
            completions,
            golds,
            data_name=self.data_name,
            num_workers=self.num_workers,
            timeout_sec=self.timeout_sec,
        )
        return [1.0 if s else 0.0 for s in scores]
