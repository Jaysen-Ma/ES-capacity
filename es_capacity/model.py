"""Base LLM loading and parameter-space noise helpers.

ES methods explore in weight space: perturb → forward/evaluate → restore
(or reconstruct from seeds). No HF / distributed wiring yet.
"""

from __future__ import annotations

from typing import Any


def load_model(model_name: str) -> Any:
    """Load a pretrained base LLM for ES post-training / evaluation.

    # TODO: load weights, tokenizer, generation config.
    """
    raise NotImplementedError(f"load_model({model_name!r})")


def apply_noise(model: Any, noise: Any) -> None:
    """Apply a parameter-space perturbation in place.

    # TODO: full-rank (Qiu) or low-rank EGGROLL (σ E) application.
    """
    raise NotImplementedError("apply_noise")


def restore_noise(model: Any, noise: Any) -> None:
    """Undo a previously applied parameter-space perturbation in place.

    # TODO: subtract the same noise (or reconstruct from seed) layer-wise.
    """
    raise NotImplementedError("restore_noise")


def generate(model: Any, prompts: list[str], **gen_kwargs: Any) -> list[str]:
    """Sample completions for capacity / fitness evaluation.

    # TODO: nucleus sampling with EvalConfig temperature / top_p / max tokens.
    """
    raise NotImplementedError("generate")
