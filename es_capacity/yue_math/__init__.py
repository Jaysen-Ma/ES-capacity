"""Yue et al. math answer extraction and grading (vendored).

Source: limit-of-RLVR ``math/examples/math_eval/{parser,grader}.py``
(Does Reinforcement Learning Really Incentivize Reasoning Capacity…,
arXiv:2504.13837).
"""

from es_capacity.yue_math.grader import math_equal
from es_capacity.yue_math.parser import extract_answer, parse_ground_truth, strip_string

__all__ = [
    "extract_answer",
    "parse_ground_truth",
    "strip_string",
    "math_equal",
]
