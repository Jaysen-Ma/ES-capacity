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

*(pending — pass@k curve for base vs. ES-trained goes here once the run and
eval complete)*

## Later

A third arm testing EGGROLL (Sarkar et al., arXiv:2511.16652 — rank-r
LoRA-factorized ES, as opposed to `es-at-scale`'s full-rank ES) is planned
once this first experiment is done.
