"""EGGROLL low-rank Perturber (v1).

The ES update follows arXiv:2511.16652 Eq. 3-6 directly (the "EGGROLL UPDATE" box,
p.7), not `third_party/eggroll/*.py` (GPL-3.0 — porting it would make this file a
derivative; see .notes/DECISIONS.md 2026-08-05 and AGENTS.md). Per the paper:

    E_i = (1/sqrt(r)) A_i B_i^T,  A_i, B_i unit-variance i.i.d.
    M_{t+1} = M_t + (alpha_t / N) * sum_i E_i * f(W = M_t + sigma*E_i)

with "the constant 1/sigma absorbed into the tunable learning rate alpha_t" -- i.e.
the update uses the *same* sigma-scaled E_i as the generation-time perturbation, not
E_i/sigma. `lora_update_term` below already encodes this (sigma baked into noise_a/
noise_b), so callers must not divide by sigma again.

Antithetic pairing is not in the paper's Eq. 6 (N_workers there is the full
population) but is a standard, literature-grounded addition (Salimans et al. 2017,
which the paper says it adapts): shaping fitness across the full population first,
then differencing pairs, is mathematically identical to using per-member shaped
fitness directly in Eq. 6, since sum_i E_i f_i = sum_pairs E_k (f_2k - f_2k+1) when
member 2k+1's perturbation is the negation of member 2k's.

Multi-LoRA on sm_121 must be probed before calling evaluate().
"""

from __future__ import annotations

import math
import shutil
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Every nn.Linear EGGROLL perturbs. Biases, embeddings, lm_head and norms are
# left untouched -- standard LoRA convention (bias="none").
TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def module_shapes(model_cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Return {module: (in_features, out_features)} for one decoder layer, from
    the model's config.json. Shared by the throughput probe and the real trainer
    so shape logic isn't duplicated."""
    hidden = int(model_cfg["hidden_size"])
    inter = int(model_cfg["intermediate_size"])
    n_heads = int(model_cfg["num_attention_heads"])
    n_kv = int(model_cfg.get("num_key_value_heads", n_heads))
    head_dim = int(model_cfg.get("head_dim", hidden // n_heads))
    q_out = n_heads * head_dim
    kv_out = n_kv * head_dim
    return {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }


@dataclass
class EngineConfig:
    """Training-time vLLM engine settings. Deliberately separate from the eval
    engine's [vllm] config (configs/machine/<host>.toml) -- changing that would
    make new eval shards incomparable with completed base/GRPO ones."""

    max_model_len: int = 6144
    max_num_seqs: int = 256
    max_loras: int = 16
    gpu_memory_utilization: float = 0.85
    dtype: str = "bfloat16"


@dataclass
class EggrollConfig:
    population_size: int = 64  # even; antithetic pairs
    lora_r: int = 1
    sigma: float = 0.001
    learning_rate: float = 0.001
    normalize_with_std: bool = False
    fitness_shaping: str = "centered_rank"  # "centered_rank" | "mean_std" | "none"
    base_seed: int = 0
    prompt_batch_size: int = 16
    max_tokens: int = 4096
    temperature: float = 0.0
    # Fresh noise every step is now free (no engine rebuild), so the default is 1.
    steps_per_adapter: int = 1
    save_freq: int = 20
    # How many materialized checkpoints to keep on disk (oldest pruned first).
    # Each is ~15GB; this is a safety/disk-space knob, not a correctness one --
    # resume never depends on old checkpoints existing, only on records.jsonl.
    checkpoint_keep: int = 20
    train_dataset: str = "math_lvl3to5"
    update_timeout_sec: float = 900.0
    target_modules: tuple[str, ...] = TARGET_MODULES
    engine: EngineConfig = field(default_factory=EngineConfig)


def get_rng_noise(
    base_seed: int,
    num_pop_pairs: int,
    pop_pair_idx: int,
    num_layers: int,
    layer_idx: int,
    step: int,
    shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seed-regenerated unit-variance (A, B), matching the paper's p(A_i), p(B_i).

    `step` here is always a `pop_step` (= es_step // steps_per_adapter): the same
    noise is reused for every ES step within one adapter block, per steps_per_adapter.
    Deterministic in (base_seed, pop_step, pair_idx, layer_idx) -- this is what makes
    resume free (no RNG state to persist, just the scalar step counter).
    """
    noise_id = base_seed + (num_pop_pairs * num_layers * step) + (pop_pair_idx * num_layers) + layer_idx
    gen = torch.Generator()
    gen.manual_seed(int(noise_id) % (2**63 - 1))
    shape_a, shape_b = shapes
    noise_a = torch.normal(mean=0.0, std=1.0, size=shape_a, generator=gen)
    noise_b = torch.normal(mean=0.0, std=1.0, size=shape_b, generator=gen)
    return noise_a, noise_b


def scaled_pair_noise(
    base_seed: int,
    num_pairs: int,
    pair_idx: int,
    num_layers: int,
    layer_idx: int,
    pop_step: int,
    lora_r: int,
    sigma: float,
    in_out_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(A, B) for one pair/layer, sigma-scaled so B @ A == sigma * E_i (paper Eq. 6).

    `in_out_shape` is `(in_features, out_features)` -- module_shapes()'s convention,
    *not* the tensor storage order. Returns A: [r, in], B: [out, r], so B @ A has
    shape [out, in], matching how PyTorch/safetensors actually store the weight.

    Splits sigma as sqrt(sigma) on A and sqrt(sigma/r) on B rather than as one
    combined (sigma/sqrt(r)) scalar on unit-variance A,B -- algebraically identical
    (A'B'^T = sigma/sqrt(r) * A_unit B_unit^T either way) but lets each factor be
    generated and scaled independently. Shared by generation-time adapter writing,
    the update term, and the checkpoint merge, so all three agree by construction.
    """
    in_f, out_f = in_out_shape
    shapes = ((lora_r, in_f), (out_f, lora_r))
    a, b = get_rng_noise(base_seed, num_pairs, pair_idx, num_layers, layer_idx, pop_step, shapes)
    a = a * math.sqrt(sigma)
    b = b * math.sqrt(sigma / lora_r)
    return a, b


class EggrollPerturber:
    """Low-rank antithetic ES via LoRA-factorized noise.

    Call attach_run() once before the first evaluate()/update() to give the
    perturber the model architecture and run directory it needs to materialize
    adapters and merge checkpoints. ESLoop drives evaluate()/update() per step;
    call finalize() after the loop to flush a partial final block.
    """

    def __init__(self, cfg: EggrollConfig | None = None):
        self.cfg = cfg or EggrollConfig()
        if self.cfg.population_size % 2 != 0:
            raise ValueError("population_size must be even (antithetic pairs)")
        self._base_model_path: Path | None = None
        self._model_cfg: dict[str, Any] = {}
        self._num_layers: int = 0
        self._shapes: dict[str, tuple[int, int]] = {}
        self._run_dir: Path | None = None
        self._adapter_root: Path | None = None
        self._ckpt_dir: Path | None = None
        self._train_engine: Any = None
        self._current_pop_step: int | None = None
        self._pair_adapters: dict[int, tuple[Path, Path]] = {}
        # Every ES record ever applied in this process. Together with base_seed
        # this reconstructs the full trajectory, so it is the real checkpoint.
        self.all_records: list[tuple[int, int, float]] = []
        self.last_diag: dict[str, Any] = {}

    @property
    def num_pairs(self) -> int:
        return self.cfg.population_size // 2

    @property
    def ckpt_dir(self) -> Path:
        assert self._ckpt_dir is not None, "attach_run() not called"
        return self._ckpt_dir

    def attach_run(
        self,
        *,
        base_model_path: str | Path,
        model_cfg: dict[str, Any],
        run_dir: str | Path,
        resume_ckpt_dir: str | Path | None = None,
    ) -> None:
        """Wire up architecture + filesystem context. Must be called before the
        first evaluate()/update(). `resume_ckpt_dir` points at the checkpoint to
        continue training from (defaults to the base model, i.e. a fresh run)."""
        self._base_model_path = Path(base_model_path)
        self._model_cfg = model_cfg
        self._num_layers = int(model_cfg["num_hidden_layers"])
        self._shapes = module_shapes(model_cfg)
        self._run_dir = Path(run_dir)
        self._adapter_root = Path("/dev/shm/es_capacity_train_adapters") / self._run_dir.name
        self._ckpt_dir = Path(resume_ckpt_dir) if resume_ckpt_dir else self._base_model_path
        (self._run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def prepare_population(self, step: int) -> list[dict[str, Any]]:
        """Return metadata for each population member (pair index + sign)."""
        members = []
        for i in range(self.cfg.population_size):
            members.append(
                {
                    "pop_idx": i,
                    "pair_idx": i // 2,
                    "sign": 1 if i % 2 == 0 else -1,
                    "step": step,
                }
            )
        return members

    def lora_update_term(
        self,
        weight_shape: tuple[int, int],
        pair_idx: int,
        layer_idx: int,
        step: int,
        fitness_diff: float,
        num_layers: int,
    ) -> torch.Tensor:
        """Weighted low-rank update contribution for one antithetic pair on one
        layer: fitness_diff * (B @ A), sigma already baked into (A, B) by
        scaled_pair_noise. `step` here must be a pop_step (see get_rng_noise).
        `weight_shape` is (out_features, in_features) -- the tensor storage
        order, matching this method's original contract."""
        out_f, in_f = weight_shape
        a, b = scaled_pair_noise(
            self.cfg.base_seed, self.num_pairs, pair_idx, num_layers, layer_idx,
            step, self.cfg.lora_r, self.cfg.sigma, (in_f, out_f),
        )
        return fitness_diff * (b @ a)

    def normalize_fitnesses(self, fitnesses: list[float], *, prompt_grouped: bool = False) -> list[float]:
        """Per-population mean-center (+ optional std), returning length=population."""
        import numpy as np

        f = np.asarray(fitnesses, dtype=np.float64)
        f = f - f.mean()
        if self.cfg.normalize_with_std:
            std = f.std()
            if std > 1e-8:
                f = f / std
        return f.tolist()

    def shape_fitnesses(self, fitnesses: list[float]) -> list[float]:
        """Per-population fitness shaping, dispatched by cfg.fitness_shaping.

        'centered_rank' (default): average-rank utility in [-0.5, 0.5], symmetric
        by construction so it sums to zero -- standard ES variance reduction
        (Wierstra et al. 2011 / Salimans et al. 2017, which the EGGROLL paper says
        it adapts). Chosen over mean/std because a binary reward averaged over 16
        prompts has only 17 distinct values, so raw fitness ties heavily and
        dividing by a near-zero std can blow up the update.
        """
        import numpy as np

        if self.cfg.fitness_shaping == "none":
            return list(fitnesses)
        if self.cfg.fitness_shaping == "mean_std":
            return self.normalize_fitnesses(fitnesses)
        if self.cfg.fitness_shaping != "centered_rank":
            raise ValueError(f"Unknown fitness_shaping {self.cfg.fitness_shaping!r}")

        f = np.asarray(fitnesses, dtype=np.float64)
        n = len(f)
        if n <= 1:
            return [0.0] * n
        order = f.argsort(kind="stable")
        sorted_f = f[order]
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_f[j + 1] == sorted_f[i]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2.0
            i = j + 1
        utility = ranks / (n - 1) - 0.5
        return utility.tolist()

    def antithetic_diffs(self, normalized: list[float]) -> list[float]:
        """fitness_diff = f[2k] - f[2k+1] for each pair."""
        diffs = []
        for k in range(self.num_pairs):
            diffs.append(normalized[2 * k] - normalized[2 * k + 1])
        return diffs

    def ensure_engine(self) -> None:
        """Build the one long-lived engine. Safe to call repeatedly."""
        from es_capacity.posttrain import engine as engine_mod

        if self._train_engine is not None:
            return
        self._train_engine = engine_mod.TrainEngine(
            self._ckpt_dir,
            max_model_len=self.cfg.engine.max_model_len,
            max_num_seqs=self.cfg.engine.max_num_seqs,
            max_loras=self.cfg.engine.max_loras,
            max_lora_rank=self.cfg.lora_r,
            gpu_memory_utilization=self.cfg.engine.gpu_memory_utilization,
            dtype=self.cfg.engine.dtype,
        )

    def replay_records(self, records: list[tuple[int, int, float]]) -> None:
        """Re-apply past ES records to the live weights (resume path). Equivalent
        to having run those steps, because the noise is seed-derived."""
        if not records:
            return
        self.ensure_engine()
        self._push_update(records)

    def evaluate(
        self,
        members: list[Any],
        prompts: list[str],
        *,
        golds: list[str],
        scorer: Any,
        **kwargs: Any,
    ) -> list[float]:
        """Generate + grade one ES step's rollouts, chunked member-major so at
        most cfg.engine.max_loras distinct adapters are ever in one batch. All
        members see the *same* prompt batch (common random numbers), which is what
        makes their fitnesses comparable.

        Side effect: populates self.last_diag with per-step diagnostics.
        """
        from es_capacity.posttrain import engine as engine_mod

        assert self._run_dir is not None, "attach_run() not called"
        step = int(members[0]["step"])
        pop_step = step // self.cfg.steps_per_adapter
        self.ensure_engine()
        self._ensure_adapters(pop_step)

        chunk_size = self.cfg.engine.max_loras
        n_prompts = len(prompts)
        fitnesses = [0.0] * self.cfg.population_size
        tok_counts: list[int] = []
        n_truncated = 0
        gen_wall = 0.0
        grade_wall = 0.0
        # completion text per (prompt position) across members, to detect a sigma
        # so small that every member emits the same thing (zero ES signal).
        per_prompt_texts: list[set[str]] = [set() for _ in range(n_prompts)]

        for chunk_start in range(0, self.cfg.population_size, chunk_size):
            chunk = list(range(chunk_start, min(chunk_start + chunk_size, self.cfg.population_size)))
            chunk_prompts: list[str] = []
            chunk_paths: list[Path] = []
            chunk_ids: list[int] = []
            for pop_idx in chunk:
                pair_idx = pop_idx // 2
                is_pos = pop_idx % 2 == 0
                adapter_path = self._pair_adapters[pair_idx][0 if is_pos else 1]
                lora_id = engine_mod.lora_int_id(pop_step, self.num_pairs, pair_idx, is_pos)
                chunk_prompts.extend(prompts)
                chunk_paths.extend([adapter_path] * n_prompts)
                chunk_ids.extend([lora_id] * n_prompts)

            t0 = time.time()
            outs = self._train_engine.generate_chunk(
                chunk_prompts,
                chunk_paths,
                chunk_ids,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            gen_wall += time.time() - t0

            t1 = time.time()
            for i, pop_idx in enumerate(chunk):
                member = outs[i * n_prompts : (i + 1) * n_prompts]
                texts = [o["text"] for o in member]
                for j, o in enumerate(member):
                    tok_counts.append(o["num_tokens"])
                    if o["finish_reason"] == "length":
                        n_truncated += 1
                    per_prompt_texts[j].add(o["text"])
                scores = scorer.score_batch(texts, golds)
                fitnesses[pop_idx] = sum(scores) / max(len(scores), 1)
            grade_wall += time.time() - t1

        tok_counts.sort()
        n = len(tok_counts) or 1
        pop = self.cfg.population_size
        self.last_diag = {
            "fitness_min": min(fitnesses),
            "fitness_max": max(fitnesses),
            "fitness_std": float(statistics.pstdev(fitnesses)) if len(fitnesses) > 1 else 0.0,
            "n_distinct_fitness": len(set(fitnesses)),
            # 1.0 => every member produced identical text (no ES signal at all)
            "mean_distinct_frac": sum(len(s) for s in per_prompt_texts) / (n_prompts * pop),
            "tok_mean": sum(tok_counts) / n,
            "tok_median": tok_counts[n // 2],
            "tok_p90": tok_counts[int(0.9 * (n - 1))],
            "tok_total": sum(tok_counts),
            "truncated_frac": n_truncated / n,
            "gen_wall_sec": round(gen_wall, 1),
            "grade_wall_sec": round(grade_wall, 1),
            "gen_tok_per_sec": round(sum(tok_counts) / gen_wall, 1) if gen_wall else 0.0,
        }
        return fitnesses

    def _ensure_adapters(self, pop_step: int) -> None:
        """Materialize this block's antithetic adapter pair dirs, once per pop_step."""
        from es_capacity.posttrain import engine as engine_mod

        if self._pair_adapters and pop_step == self._current_pop_step:
            return
        self._drop_adapter_dir(self._current_pop_step)
        self._pair_adapters = engine_mod.write_pair_adapters(
            self._ckpt_dir,
            self._adapter_root / f"pop_step_{pop_step:05d}",
            pair_indices=list(range(self.num_pairs)),
            num_layers=self._num_layers,
            shapes=self._shapes,
            lora_r=self.cfg.lora_r,
            sigma=self.cfg.sigma,
            base_seed=self.cfg.base_seed,
            pop_step=pop_step,
            num_pairs=self.num_pairs,
        )
        self._current_pop_step = pop_step

    def _drop_adapter_dir(self, pop_step: int | None) -> None:
        """/dev/shm is only 61 GB and each block writes population_size adapters."""
        if pop_step is None or self._adapter_root is None:
            return
        shutil.rmtree(self._adapter_root / f"pop_step_{pop_step:05d}", ignore_errors=True)

    def update(self, members: list[Any], fitnesses: list[float], step: int) -> None:
        """Shape fitness, difference the antithetic pairs, and push the resulting
        ES update straight into the live GPU weights."""
        shaped = self.shape_fitnesses(fitnesses)
        diffs = self.antithetic_diffs(shaped)
        pop_step = step // self.cfg.steps_per_adapter
        records = [(pop_step, pair_idx, d) for pair_idx, d in enumerate(diffs)]

        t0 = time.time()
        res = self._push_update(records)
        self.last_diag = dict(getattr(self, "last_diag", {}))
        self.last_diag["update_wall_sec"] = round(time.time() - t0, 1)
        self.last_diag["max_abs_delta"] = res.get("max_abs_delta", 0.0)
        self.all_records.extend(records)

    def _push_update(self, records: list[tuple[int, int, float]]) -> dict[str, float]:
        return self._train_engine.apply_es_update(
            records,
            model_cfg=self._model_cfg,
            shapes=self._shapes,
            num_layers=self._num_layers,
            num_pairs=self.num_pairs,
            lora_r=self.cfg.lora_r,
            sigma=self.cfg.sigma,
            base_seed=self.cfg.base_seed,
            scale=self.cfg.learning_rate / self.cfg.population_size,
            timeout=self.cfg.update_timeout_sec,
        )

    def save_checkpoint(self, step: int) -> Path:
        """Materialize an evaluable HF checkpoint by replaying all records onto
        the base model. The engine keeps running -- this only reads from disk."""
        from es_capacity.posttrain import engine as engine_mod

        ckpt_out = self._run_dir / "checkpoints" / f"step_{step:04d}"
        engine_mod.materialize_checkpoint(
            self._base_model_path,
            ckpt_out,
            records=self.all_records,
            num_layers=self._num_layers,
            shapes=self._shapes,
            lora_r=self.cfg.lora_r,
            sigma=self.cfg.sigma,
            base_seed=self.cfg.base_seed,
            num_pairs=self.num_pairs,
            scale=self.cfg.learning_rate / self.cfg.population_size,
        )
        self._prune_checkpoints(keep=self.cfg.checkpoint_keep)
        return ckpt_out

    def close(self) -> None:
        self._drop_adapter_dir(self._current_pop_step)
        if self._train_engine is not None:
            self._train_engine.close()
            self._train_engine = None

    def _prune_checkpoints(self, *, keep: int) -> None:
        ckpt_root = self._run_dir / "checkpoints"
        if not ckpt_root.exists():
            return
        dirs = sorted(d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("step_"))
        for d in dirs[:-keep]:
            shutil.rmtree(d, ignore_errors=True)
