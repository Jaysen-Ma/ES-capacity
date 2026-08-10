# ES-capacity

Does Evolution-Strategies post-training preserve a base model's pass@k
ceiling better than gradient-based RLVR (GRPO)? RLVR reliably raises pass@1
but tends to narrow the pass@k ceiling relative to the base model — this
project tests whether ES-based fine-tuning (gradient-free, population-based)
avoids that narrowing, on the same task/reward/data.

Training and evaluation run from forked, patched copies of the original
papers' code.

## Code

| Purpose | Repo | Branch | Notes |
|---|---|---|---|
| ES training | [Jaysen-Ma/es-at-scale](https://github.com/Jaysen-Ma/es-at-scale) (fork of [VsonicV/es-at-scale](https://github.com/VsonicV/es-at-scale), arXiv:2509.24372) | `fix/multi-engine-colocation` | Full-rank ES, Ray-orchestrated multi-engine vLLM. 3 fixes needed to run on a single-node multi-GPU box: vLLM ≥0.20 import compat, co-located-engine port/cache-collision + concurrent-compile races, `--start-iteration` resume support. Verified on 8x RTX 4090. |
| pass@k generation, grading, plotting | [Jaysen-Ma/limit-of-RLVR](https://github.com/Jaysen-Ma/limit-of-RLVR) (fork of [LeapLabTHU/limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR), arXiv:2504.13837) | `fix/math-equal-timeout-bypass` | `math/examples/math_eval/` already implements the unbiased pass@k estimator and a full generate+grade harness (`math_eval.py`, `--n_sampling`). Fixes a real grading-worker hang (`math_equal`'s equation-comparison branch bypassed the timeout guard). |

Both forks' `main` are kept as unmodified mirrors of upstream; all changes
live on the branches above so they can be sent upstream as PRs later.

## Experiment 1: Qwen2.5-1.5B base vs. ES-at-scale-trained

| Param | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B` (base, not Instruct) |
| Task | math |
| sigma | 0.001 |
| alpha (lr) | auto (`sigma/2` = 0.0005, `es-at-scale`'s default when `--alpha` is unset) |
| Population size | 32 |
| Iterations | 50 |
| Train dataset | `math_lvl3to5_8k` (matches SimpleRL-Zoo's training set) |
| Batch size / mini-batch size | 256 / 256 |
| Max tokens | 2048 |
| vLLM engines | 8 (one per GPU) |
| GPUs | 0-7 (8x RTX 4090 48GB) |
| Training wall-clock | 3h 22m 35s (2026-08-10 12:51:06 → 16:13:41 UTC), single uninterrupted run, includes the iteration-0 and final-iteration eval passes |

Train:
```bash
cd es-at-scale  # Jaysen-Ma/es-at-scale @ fix/multi-engine-colocation
python -m es_at_scale.train \
  --task math \
  --model-name Qwen/Qwen2.5-1.5B \
  --sigma 0.001 \
  --population-size 32 \
  --n-iterations 50 \
  --train-dataset datasets/train/math_lvl3to5_8k \
  --batch-size 256 --mini-batch-size 256 \
  --max-tokens 2048 \
  --n-vllm-engines 8 --use-gpus 0,1,2,3,4,5,6,7
```

Evaluate (base and the final ES checkpoint, same command against each;
temperature/top_p/n match this project's prior pass@64 convention):
```bash
cd limit-of-RLVR/math/examples/math_eval  # Jaysen-Ma/limit-of-RLVR @ fix/math-equal-timeout-bypass
python math_eval.py \
  --data_names minerva_math \
  --model_name_or_path <path-to-checkpoint-or-Qwen/Qwen2.5-1.5B> \
  --prompt_type qwen-boxed \
  --n_sampling 64 \
  --temperature 0.6 --top_p 0.95 \
  --max_tokens_per_call 2048 \
  --use_vllm --save_outputs
```

## Results

Minerva Math, `n_sampling=64`, identical generation settings for both models
(`temperature=0.6`, `top_p=0.95`, `max_tokens=2048`, `qwen-boxed` template,
`seed=1`; enforced by a single wrapper script so the two runs can't drift —
see `scripts/run_base_vs_trained_eval.sh`).

**Unlike the RLVR pattern the source paper documents (RL narrows the pass@k
ceiling, base models catch up and overtake at large k), the ES-trained model
stays *above* the base model at every k from 1 to 64** — no crossover:

| k | Base | ES-trained |
|---|---|---|
| 1 | 1.75% | 2.75% |
| 2 | 3.32% | 5.09% |
| 4 | 6.05% | 8.95% |
| 8 | 10.33% | 14.59% |
| 16 | 16.16% | 21.58% |
| 32 | 22.97% | 28.94% |
| 64 | 30.15% | 36.40% |

![pass@k curve](results/iter50/minerva_math_passk.png)

Four-way solvable/unsolvable breakdown (272 Minerva questions, "solvable" =
at least 1 of 64 samples correct):

| | Trained solves | Trained fails |
|---|---|---|
| **Base solves** | 24.3% | 5.9% |
| **Base fails** | 12.1% | 57.7% |

12.1% of questions gained (base couldn't solve, ES-trained can) against 5.9%
lost (base could, ES-trained can't) — a net +6.2 points of question coverage,
with about 2x as many gains as losses. Combined with the pass@k curve never
crossing back below base, this is consistent with genuine capacity expansion
rather than the sampling-efficiency-for-ceiling tradeoff RLVR typically shows.

Caveat: `n_sampling=64` is well below the source paper's own budget for this
kind of claim (k up to 1024 for AIME, 128 for MATH500), so the tail of the
curve (k>32) carries more estimator variance than the head — the qualitative
"no crossover" finding is the robust part of this result, not the exact
percentages at k=64.

Raw generations: `results/iter50/{base,trained}/minerva_math/` (gitignored,
~85MB each — regenerate with `scripts/run_base_vs_trained_eval.sh`).

## Later

A third arm testing EGGROLL (Sarkar et al., arXiv:2511.16652 — rank-r
LoRA-factorized ES, as opposed to `es-at-scale`'s full-rank ES) is planned
once this first experiment is done.
