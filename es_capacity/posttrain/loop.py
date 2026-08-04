"""ES training loop orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from es_capacity.posttrain.scorer import YueMathScorer


class ESLoop:
    def __init__(
        self,
        perturber: Any,
        scorer: YueMathScorer | None = None,
        *,
        output_dir: str | Path,
        eval_fn: Callable[[], dict[str, float]] | None = None,
        eval_every: int = 5,
    ):
        self.perturber = perturber
        self.scorer = scorer or YueMathScorer()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_fn = eval_fn
        self.eval_every = eval_every
        self.history: list[dict[str, Any]] = []

    def step(
        self,
        step: int,
        prompts: list[str],
        golds: list[str],
    ) -> dict[str, float]:
        members = self.perturber.prepare_population(step)
        # evaluate() must return one fitness per member (mean reward over prompts)
        fitnesses = self.perturber.evaluate(members, prompts, golds=golds, scorer=self.scorer)
        self.perturber.update(members, fitnesses, step)
        mean_r = float(sum(fitnesses) / max(len(fitnesses), 1))
        row = {"step": step, "mean_reward": mean_r, "wall": time.time()}
        if self.eval_fn and step % self.eval_every == 0:
            row["eval"] = self.eval_fn()
        self.history.append(row)
        (self.output_dir / "history.jsonl").open("a").write(json.dumps(row) + "\n")
        return {"mean_reward": mean_r}

    def reward_rising(self, window: int = 5) -> bool:
        if len(self.history) < window + 1:
            return False
        early = sum(h["mean_reward"] for h in self.history[:window]) / window
        late = sum(h["mean_reward"] for h in self.history[-window:]) / window
        return late > early + 1e-4

    def save_meta(self) -> None:
        cfg = getattr(self.perturber, "cfg", None)
        meta = {"history_len": len(self.history)}
        if cfg is not None and is_dataclass(cfg):
            meta["cfg"] = asdict(cfg)
        (self.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
