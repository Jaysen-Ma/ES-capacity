"""Load prompts / verifiable reasoning tasks.

AIME24 is loaded from Hugging Face (`HuggingFaceH4/aime_2024`).
Prompting follows Yue et al. ``qwen-boxed`` (limit-of-RLVR).
"""

from __future__ import annotations

from typing import Any, Iterable

AIME24_HF_ID = "HuggingFaceH4/aime_2024"

# Yue / SimpleRL ``qwen-boxed`` template (applied to both Base and Instruct).
QWEN_BOXED_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{problem}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def load_dataset(name: str, split: str = "test") -> list[dict[str, Any]]:
    """Load a verifiable-task dataset by short name."""
    name = name.lower().replace("-", "").replace("_", "")
    if name in {"aime24", "aime2024"}:
        return load_aime24()
    raise ValueError(f"Unknown dataset {name!r} (split={split!r})")


def load_aime24() -> list[dict[str, Any]]:
    """Load AIME 2024 (30 problems) and normalize records.

    Source: https://huggingface.co/datasets/HuggingFaceH4/aime_2024
    """
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset(AIME24_HF_ID, split="train")
    records: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        problem = row.get("problem") or row.get("question") or ""
        answer = row.get("answer")
        if answer is None:
            raise KeyError(f"AIME24 row {i} missing 'answer'")
        records.append(
            {
                "id": str(row.get("id", i)),
                "problem": str(problem).strip(),
                "answer": str(answer).strip(),
            }
        )
    if len(records) == 0:
        raise RuntimeError(f"Empty dataset from {AIME24_HF_ID}")
    return records


def build_prompt(
    example: dict[str, Any],
    tokenizer: Any = None,
    *,
    use_chat_template: bool = False,
) -> str:
    """Build Yue ``qwen-boxed`` prompt for one AIME example.

    Both Base and Instruct use the same string template (Yue zero-shot setup).
    ``use_chat_template`` is ignored; kept for call-site compatibility.
    """
    del tokenizer, use_chat_template
    return QWEN_BOXED_TEMPLATE.format(problem=example["problem"].strip())


def iter_prompts(dataset: Iterable[dict[str, Any]]) -> Iterable[str]:
    """Yield qwen-boxed prompts."""
    for example in dataset:
        yield build_prompt(example)


def should_use_chat_template(model_path: str, tokenizer: Any = None) -> bool:
    """Deprecated for AIME eval — Yue uses the same qwen-boxed string for both."""
    del model_path, tokenizer
    return False
