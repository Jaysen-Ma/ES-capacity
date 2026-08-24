#!/usr/bin/env python3
"""Expose the ES arm's grader to verl as a custom reward function.

The GRPO arm must optimize the SAME objective the ES arm did. verl's built-in
`hf_math_verify` is materially more permissive than ES's grader: its `qwen_extract_answer`
falls back through "the answer is" -> "final answer is" -> THE LAST NUMBER IN THE STRING,
so it scores responses that never emit `\\boxed{}` at all. ES's `boxed_reward_fn` returns
0.0 for those. Using verl's default would hand GRPO an easier reward and erase the format
headroom ES had to climb.

Wire up with:
    custom_reward_function.path=/workspace/es-capacity/scripts/es_reward_verl.py
    custom_reward_function.name=compute_score
"""
import os
import sys

ES_AT_SCALE = os.environ.get("ES_AT_SCALE_PATH", "/workspace/repos/es-at-scale")
if ES_AT_SCALE not in sys.path:
    sys.path.insert(0, ES_AT_SCALE)

from es_at_scale.reward_function.math_grader import boxed_reward_fn  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl reward hook. Returns binary {0.0, 1.0}: correctness only.

    No format reward, no format penalty, no truncation penalty — identical to the ES arm.
    A response that hits the 2048-token cap simply lacks a closed \\boxed{} and scores 0.0,
    which is exactly what happened on the ES side.
    """
    try:
        _meta, score = boxed_reward_fn(solution_str, ground_truth)
        return float(score)
    except Exception:
        # ES's grader can raise on pathological sympy input. It treats those as unparseable,
        # i.e. reward 0.0 — match that rather than crashing the training step.
        return 0.0
