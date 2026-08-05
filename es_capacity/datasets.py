"""Dataset loading for Minerva (and later Olympiad / AIME)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from es_capacity.config import AppConfig, REPO_ROOT

# Yue et al. qwen-boxed template (from limit-of-RLVR utils.PROMPT_TEMPLATES)
QWEN_BOXED = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

TEMPLATES = {
    "qwen-boxed": QWEN_BOXED,
}


def dataset_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def minerva_path(cfg: AppConfig | None = None) -> Path:
    if cfg is not None:
        return cfg.data_dir / "minerva_math" / "test.jsonl"
    return REPO_ROOT / "data" / "minerva_math" / "test.jsonl"


def load_minerva_math(
    path: Path | None = None,
    *,
    num_problems: int | None = None,
) -> list[dict[str, Any]]:
    path = path or minerva_path()
    examples: list[dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            if "idx" not in ex:
                ex["idx"] = i
            examples.append(ex)
    examples.sort(key=lambda x: x["idx"])
    if num_problems is not None and num_problems > 0:
        examples = examples[:num_problems]
    return examples


def simplerl_path(cfg: AppConfig | None = None, split: str = "math_lvl3to5") -> Path:
    root = cfg.data_dir if cfg is not None else REPO_ROOT / "data"
    return root / f"simplerl_{split}" / "train.jsonl"


def load_simplerl_math(
    path: Path | None = None,
    *,
    split: str = "math_lvl3to5",
    num_problems: int | None = None,
    cfg: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Load the GRPO baseline's training set (see cli.fetch_train_data).

    Each record carries `prompt` verbatim from SimpleRL-Zoo, so ES training
    sees exactly the prompt GRPO was trained on, and `answer` as the gold
    string (already extracted upstream — no ground-truth parsing needed).
    """
    path = path or simplerl_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run: python -m es_capacity.cli.fetch_train_data --split {split}"
        )
    examples = [json.loads(line) for line in path.open() if line.strip()]
    if num_problems is not None and num_problems > 0:
        examples = examples[:num_problems]
    return examples


def build_prompt(example: dict[str, Any], template: str = "qwen-boxed") -> str:
    if template not in TEMPLATES:
        raise KeyError(f"Unknown template {template!r}; known={list(TEMPLATES)}")
    question = example.get("problem") or example.get("question") or ""
    return TEMPLATES[template].format(input=question)


def gold_from_example(example: dict[str, Any], data_name: str = "minerva_math") -> str:
    """Parse ground-truth answer using vendored Yue parser when available."""
    import sys

    yue = REPO_ROOT / "third_party" / "yue_math"
    if str(yue) not in sys.path:
        sys.path.insert(0, str(yue))
    from parser import parse_ground_truth  # type: ignore

    _cot, ans = parse_ground_truth(example, data_name)
    return ans
