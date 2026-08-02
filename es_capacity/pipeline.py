"""End-to-end capacity experiment skeleton.

train ES → evaluate pass@k (base vs ES) → compare capacity / Δ_SE.
"""

from __future__ import annotations

from typing import Any

from es_capacity.config import ExperimentConfig
from es_capacity.data import load_dataset
from es_capacity.eval.capacity import compare_capacity
from es_capacity.eval.passk import evaluate_passk
from es_capacity.model import load_model


def train_es(base_model: Any, cfg: ExperimentConfig) -> Any:
    """Post-train `base_model` with the configured ES method.

    Dispatches to Qiu full-parameter ES or Sarkar EGGROLL.
    # TODO: construct trainer from cfg.es.method and run cfg.es.num_iterations.
    """
    raise NotImplementedError(f"train_es(method={cfg.es.method!r})")


def run_capacity_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """Run the base-vs-ES reasoning-capacity comparison.

    Outline:
      base = load_model(cfg.base_model)
      es_model = train_es(base, cfg)
      curves = {
          "base": evaluate_passk(base, data, cfg.eval.ks),
          "es": evaluate_passk(es_model, data, cfg.eval.ks),
      }
      return compare_capacity(curves)
    """
    _ = load_model  # named in the outline above
    _ = load_dataset
    _ = evaluate_passk
    _ = compare_capacity
    raise NotImplementedError("run_capacity_experiment")
