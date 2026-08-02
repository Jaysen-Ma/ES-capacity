"""pass@k estimation for reasoning-capacity boundaries (Yue et al.).

A problem counts as solved if any of k samples passes the verifier.
Use the unbiased estimator over n ≥ k samples with c correct.
"""

from __future__ import annotations

from typing import Any


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator for one problem.

    Given n samples of which c are correct, estimate Prob[at least one
    correct in k draws] without replacement bias.

    # TODO: implement 1 - C(n - c, k) / C(n, k) (with edge cases).
    """
    raise NotImplementedError(f"estimate_pass_at_k(n={n}, c={c}, k={k})")


def evaluate_passk(
    model: Any,
    dataset: list[dict[str, Any]],
    ks: list[int],
    *,
    n_samples: int | None = None,
    gen_kwargs: dict[str, Any] | None = None,
) -> dict[int, float]:
    """Average pass@k over `dataset` for each k in `ks`.

    Outline:
      for each problem: sample n completions; count correct via verify
      return {k: mean_i estimate_pass_at_k(n, c_i, k) for k in ks}

    # TODO: sampling loop, verification, aggregation.
    """
    raise NotImplementedError("evaluate_passk")
