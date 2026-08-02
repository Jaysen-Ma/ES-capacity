"""Load prompts / verifiable reasoning tasks.

Downstream: math / coding-style datasets with a ground-truth verifier
compatible with reward.R(x, y) ∈ {0, 1}.
"""

from __future__ import annotations

from typing import Any, Iterable


def load_dataset(name: str, split: str = "test") -> list[dict[str, Any]]:
    """Load a verifiable-task dataset.

    Each item is expected to provide at least:
      - prompt / question text
      - enough info for a deterministic verifier (answer, unit tests, ...)

    # TODO: wire concrete dataset loaders (e.g. MATH, GSM8K, coding).
    """
    raise NotImplementedError(f"load_dataset({name!r}, split={split!r})")


def iter_prompts(dataset: Iterable[dict[str, Any]]) -> Iterable[str]:
    """Yield model prompts from dataset records.

    # TODO: define prompt template(s) shared by training and evaluation.
    """
    raise NotImplementedError("iter_prompts")
