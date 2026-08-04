"""Post-training abstractions: Perturber / Scorer / Aggregator / ESLoop."""

from __future__ import annotations

from typing import Any, Protocol

from es_capacity.posttrain.scorer import YueMathScorer
from es_capacity.posttrain.eggroll import EggrollPerturber
from es_capacity.posttrain.qiu import QiuPerturber
from es_capacity.posttrain.loop import ESLoop

__all__ = [
    "Perturber",
    "YueMathScorer",
    "EggrollPerturber",
    "QiuPerturber",
    "ESLoop",
]


class Perturber(Protocol):
    def prepare_population(self, step: int) -> list[Any]: ...
    def evaluate(self, members: list[Any], prompts: list[str], **kwargs) -> list[float]: ...
    def update(self, members: list[Any], fitnesses: list[float], step: int) -> None: ...
    def save_checkpoint(self, path: str) -> None: ...
