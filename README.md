# ES-capacity

Does Evolution-Strategies post-training preserve a base model's pass@k
ceiling better than gradient-based RLVR (GRPO)? RLVR reliably raises pass@1
but tends to narrow the pass@k ceiling relative to the base model — this
project tests whether ES-based fine-tuning (gradient-free, population-based)
avoids that narrowing, on the same task/reward/data.

Run so far at two scales, with **opposite outcomes**: at 1.5B ES expands
question coverage on all 4 benchmarks with no pass@k crossover; at 7B, under
identical hyperparameters, the coverage gain nearly vanishes and 2 of 4
benchmarks cross over — the ceiling-narrowing pattern the RLVR paper
documents, now showing up in gradient-free ES. Out-of-domain, GPQA-diamond
finds one large unexplained *gain* at 1.5B and a consistent but
individually-insignificant downward drift across all three 7B ES arms
([below](#out-of-domain-check-gpqa-diamond)).

Doubling the 7B run to 100 iterations changed essentially nothing, and raising
the perturbation scale only ever made things worse — see
[Additional runs](#additional-runs-perturbation-scale-and-run-length).

The RL arm throughout is a published SimpleRL-Zoo checkpoint, which saw
**12.01 epochs of the shared 8,523-problem training set against this ES run's
1.50** ([arithmetic](#results--experiment-1-qwen25-15b)) — so read every
ES-vs-RL row as ES on an exactly 8x smaller data budget, not as a matched
comparison.

Training and evaluation run from forked, patched copies of the original
papers' code.

## Code

| Purpose | Repo | Branch | Notes |
|---|---|---|---|
| ES training | [Jaysen-Ma/es-at-scale](https://github.com/Jaysen-Ma/es-at-scale) (fork of [VsonicV/es-at-scale](https://github.com/VsonicV/es-at-scale), arXiv:2509.24372) | `fix/multi-engine-colocation` | Full-rank ES, Ray-orchestrated multi-engine vLLM. 3 fixes needed to run on a single-node multi-GPU box: vLLM ≥0.20 import compat, co-located-engine port/cache-collision + concurrent-compile races, `--start-iteration` resume support. Verified on 8x RTX 4090. |
| pass@k generation, grading, plotting | [Jaysen-Ma/limit-of-RLVR](https://github.com/Jaysen-Ma/limit-of-RLVR) (fork of [LeapLabTHU/limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR), arXiv:2504.13837) | `fix/math-equal-timeout-bypass` | `math/examples/math_eval/` already implements the unbiased pass@k estimator and a full generate+grade harness (`math_eval.py`, `--n_sampling`). Fixes a real grading-worker hang (`math_equal`'s equation-comparison branch bypassed the timeout guard). |

Both forks' `main` are kept as unmodified mirrors of upstream; all changes
live on the branches above so they can be sent upstream as PRs later.

## Experiment 1: Qwen2.5-1.5B base vs. ES-at-scale-trained (done)

| Param | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B` (base, not Instruct) |
| Task | math |
| sigma | 0.001 |
| alpha (lr) | auto (`sigma/2` = 0.0005, `es-at-scale`'s default when `--alpha` is unset) |
| Population size | 32 |
| Iterations | 50 |
| Train dataset | `math_lvl3to5_8k` — the same 8,523 problems, in the same order, as SimpleRL-Zoo's `simplelr_qwen_level3to5` split (verified against their released parquet) |
| Batch size / mini-batch size | 256 / 256 |
| Max tokens | 2048 |
| vLLM engines | 8 (one per GPU) |
| GPUs | 0-7 (8x RTX 4090 48GB) |
| Training wall-clock | 3h 22m 35s (including 2 evals) |

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
  --eval-dataset datasets/evaluation_suite/math \
  --batch-size 256 --mini-batch-size 256 \
  --max-tokens 2048 \
  --n-vllm-engines 8 --use-gpus 0,1,2,3,4,5,6,7
```
`--eval-dataset` matters even though this README's reported pass@k numbers
come from a separate eval pass (below), not the trainer's own in-loop eval —
omit it and it silently defaults to the `countdown` task's eval set, which
has no `problem` field and crashes the trainer the moment anything touches
the eval dataloader (e.g. resuming training from a checkpoint, see Experiment
2 below).

Evaluate (base, ES-trained, and any third arm, all through identical
settings, sharded across every GPU — `n_sampling` = 512 for AIME24, 128 for
the rest):
```bash
cd ES-capacity
scripts/run_base_vs_trained_eval.sh <path-to-hf-checkpoint> 1.5b-sigma001-iter50 aime24 512
scripts/run_base_vs_trained_eval.sh <path-to-hf-checkpoint> 1.5b-sigma001-iter50 math500 128
scripts/run_base_vs_trained_eval.sh <path-to-hf-checkpoint> 1.5b-sigma001-iter50 minerva_math 128
scripts/run_base_vs_trained_eval.sh <path-to-hf-checkpoint> 1.5b-sigma001-iter50 olympiadbench 128
# or all 4 at once:
scripts/run_full_eval_suite.sh <path-to-hf-checkpoint> 1.5b-sigma001-iter50

# third-arm (e.g. an RL baseline), reusing the base model's already-computed outputs:
scripts/run_third_model_full_suite.sh <third-model-dir> rl 1.5b-sigma001-iter50 [n_sampling_override]
```
`convert_to_hf.py` first, if starting from a raw `es-at-scale` checkpoint —
see [Model](#model) below. Both wrappers take an optional trailing
`base_model_dir` (added for Experiment 2); it defaults to the Qwen2.5-1.5B
snapshot used here.

### Evaluation wall-clock

Per-benchmark generation time (base, ES-trained, RL), 8x RTX 4090 48GB,
sharded across all 8 GPUs, `n_sampling`/model/benchmark as above:

| Benchmark | Base | ES-trained | RL |
|---|---|---|---|
| AIME24 (n=512) | 5m23s | 4m34s | 5m24s |
| MATH500 (n=128) | 3m38s | 2m40s | 3m18s |
| Minerva (n=128) | 4m32s | 4m08s | 5m05s |
| OlympiadBench (n=128) | 5m54s | 9m41s | 7m33s |
| **Total** | **19m27s** | **21m02s** | **21m20s** |

Grand total across all 3 models, 4 benchmarks: ~1h2m.

## Results — Experiment 1 (Qwen2.5-1.5B)

Three models compared on four benchmarks (AIME24, MATH500, Minerva Math,
OlympiadBench) at identical generation settings throughout (`temperature=0.6`,
`top_p=0.95`, `max_tokens=2048`, `qwen-boxed` template, `seed=1` — enforced by
shared wrapper scripts so runs can't drift: `scripts/run_base_vs_trained_eval.sh`,
`scripts/run_third_model_eval.sh`). `n_sampling` (= max k) is 512 for AIME24,
128 for the rest; each benchmark's full data range gets its own pass@k curve.

- **Base**: `Qwen/Qwen2.5-1.5B`
- **ES**: this project's Experiment 1 checkpoint
- **RL**: [hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo),
  a real published GRPO-trained model, same base and (matched) training data

<table>
<tr>
<td><img src="results/1.5b-sigma001-iter50/aime24_threeway_passk.png" width="390" alt="AIME24 pass@k"></td>
<td><img src="results/1.5b-sigma001-iter50/math500_threeway_passk.png" width="390" alt="MATH500 pass@k"></td>
</tr>
<tr>
<td><img src="results/1.5b-sigma001-iter50/minerva_math_threeway_passk.png" width="390" alt="Minerva pass@k"></td>
<td><img src="results/1.5b-sigma001-iter50/olympiadbench_threeway_passk.png" width="390" alt="OlympiadBench pass@k"></td>
</tr>
</table>

Per-k numbers for all three arms: **[results/README.md](results/README.md)**.

**ES vs. base: no crossover on any of the 4 benchmarks** — ES stays above
base across the entire k range tested, unlike the pass@k-ceiling-narrowing
pattern the source RLVR paper documents for gradient-based methods. (This is
the finding [Experiment 2](#results--experiment-2-7b) fails to reproduce at
7B.)
Solvable/unsolvable breakdown ("solvable" = at least 1 of `n_sampling`
completions correct) confirms this nets positive everywhere, gains
consistently outweighing losses:

| Benchmark | ES: narrow / gain / **net** | RL: narrow / gain / **net** |
|---|---|---|
| AIME24 | 0.0% / 13.3% / **+13.3** | 3.3% / 30.0% / **+26.7** |
| MATH500 | 1.4% / 12.0% / **+10.6** | 1.2% / 12.6% / **+11.4** |
| Minerva | 7.0% / 10.3% / **+3.3** | 5.5% / 12.5% / **+7.0** |
| OlympiadBench | 3.9% / 13.3% / **+9.4** | 3.4% / 12.9% / **+9.5** |

**ES vs. RL is a mixed picture, not a clean win for either.** RL narrows
*less* than ES on 3 of 4 benchmarks and matches or exceeds ES's net gain on
all 4. No consistent crossover pattern either:

- **AIME24**: RL and ES are essentially tied through k≈16, then RL's lead
  *grows* with k, reaching pass@512 = 50.0% vs ES's 36.7% (n=30 questions
  here, so the noisiest of the four benchmarks).
- **MATH500**: RL starts *below* ES at low k (pass@1: 13.4% vs 16.8%),
  crosses over around k≈16-32, finishes slightly ahead at k=128 (90.0% vs
  89.2%).
- **Minerva**: RL stays above ES at every k from 1 to 128, gap plateauing
  around +3.5 to +4.4 points by k≥16.
- **OlympiadBench**: the two stay close throughout (within ±1 point),
  converging to a near-exact tie by k=128.

**Not a compute-matched comparison — the data budgets differ by exactly 8x.**
Both arms train on the same **8,523 problems** (SimpleRL-Zoo's
`simplelr_qwen_level3to5` split; verified identical, same order). They pass
over it a very different number of times:

| | Prompts/step | Steps | Prompt-exposures | **Epochs over the 8,523** | Generations |
|---|---|---|---|---|---|
| **ES** (this run) | 256 | 50 | 12,800 | **1.50** | 409,600 (x32 population) |
| **RL** (SimpleRL-Zoo 1.5B) | 1,024 | ~100 | 102,400 | **12.01** | 819,200 (x8 rollouts) |

SimpleRL-Zoo's published recipe is "a prompt batch size of 1,024 and 8 rollouts
per prompt" (arXiv:2503.18892 §B.5), and its Qwen2.5-1.5B training curves run
to ~100 steps — so the released checkpoint has made **12.01 passes over the
data where this ES run made 1.50**, at 2x the raw generation count. The
exposure ratio is exactly 8.00x and does not depend on the dataset size.

Two caveats on those figures. The step count is read off their per-model
figures, since the paper never states RL epochs directly. And the paper
describes the split as "approximately 8,000 problems" (§3.2) — dividing by that
round number is where the commonly-quoted **12.8** epochs comes from; against
the actual 8,523-row file it is 12.01, and the matching ES figure moves 1.60 →
1.50. Either pairing gives the same 8.00x.

That reframes the ES-vs-RL rows above: RL matching or slightly beating ES here
is what an 8x larger data budget should buy. It doesn't rule out ES preserving
capacity better under matched compute — that comparison still hasn't been run.
What holds regardless, at this scale: ES itself shows no crossover and a clean
net-positive result relative to the base model on all 4 benchmarks, off 1.50
epochs.

## Experiment 2: Qwen2.5-7B base vs. ES-at-scale-trained (done)

Same task, same reward, same data, same ES hyperparameters as Experiment 1 —
**only the base model changes** (1.5B → 7B). No retuning turned out to be
needed on the same 8x RTX 4090 48GB box. The point is to test whether
Experiment 1's clean capacity-expansion result is a property of ES or a
property of a small, weak base model with a lot of headroom.

| Param | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B` (base, not Instruct) |
| Task | math |
| sigma | 0.001 |
| alpha (lr) | auto (`sigma/2` = 0.0005, `--alpha -1`) |
| Population size | 32 |
| Iterations | 50 (`--n-iterations 50`; the trainer actually runs 51 — an off-by-one in `es-at-scale`'s loop that applies to Experiment 1 too) |
| Train dataset | `math_lvl3to5_8k` (8,523 problems, as Experiment 1) |
| Batch size / mini-batch size | 256 / 256 |
| Max tokens | 2048 |
| Seed | 42 |
| vLLM engines | 8 (one per GPU) |
| GPUs | 0-7 (8x RTX 4090 48GB) |
| Training wall-clock | 4h 35m 33s (~300s/iteration, near-constant; includes the baseline and final eval passes the training script runs itself) |
| W&B | none (`--logging none`) — per-iteration stats come from the trainer's stdout, kept in the `es-at-scale` working tree, not this repo |

Train:
```bash
cd es-at-scale  # Jaysen-Ma/es-at-scale @ fix/multi-engine-colocation
python -m es_at_scale.train \
  --task math \
  --model-name Qwen/Qwen2.5-7B \
  --experiment-name qwen7b-math-run \
  --sigma 0.001 \
  --population-size 32 \
  --n-iterations 50 \
  --train-dataset datasets/train/math_lvl3to5_8k \
  --eval-dataset datasets/evaluation_suite/math \
  --batch-size 256 --mini-batch-size 256 \
  --max-tokens 2048 \
  --n-vllm-engines 8 --use-gpus 0,1,2,3,4,5,6,7
```
See the `--eval-dataset` note under Experiment 1 above — it's easy to forget
since this run doesn't crash without it (only a later `--checkpoint` resume
does), but the trainer's in-loop eval would still silently be scoring the
wrong task.

Evaluate — same wrappers and same fixed settings as Experiment 1, with the 7B
base passed explicitly as the trailing `base_model_dir`:
```bash
cd ES-capacity
scripts/run_full_eval_suite.sh \
  /workspace/es-at-scale/experiments/qwen7b-math-run/hf-checkpoint-iter50 \
  7b-sigma001-iter50 \
  /workspace/.hf_home/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
```

### Evaluation wall-clock

Base then ES-trained, sequential per benchmark, same 8-GPU sharding. No RL arm
was run at 7B (see [Later](#later)).

| Benchmark | Gens/model | Base | ES-trained | Base gen/s | ES gen/s |
|---|---|---|---|---|---|
| AIME24 (n=512) | 15,360 | 13m41s | 13m12s | 18.7 | 19.4 |
| MATH500 (n=128) | 64,000 | 11m19s | 26m07s | 94.3 | 40.8 |
| Minerva (n=128) | 34,816 | 24m33s | 25m07s | 23.6 | 23.1 |
| OlympiadBench (n=128) | 86,400 | 40m21s | 43m17s | 35.7 | 33.3 |
| **Total generation** | | **1h30m** | **1h48m** | | |

Full suite wall-clock (launch → "All benchmarks complete"): **≈3h35m**; the
~18min over the 3h17m of generation is model-loading/private-copy overhead
across the 8 base/trained loads.

**The ES-trained model generates ~2.3x slower on MATH500 and at parity on the
other three.** Working theory: ES makes it markedly more accurate on the easy
benchmark (46.8% → 71.4% on the training suite's own eval) and it now spends
more tokens per answer on problems it used to answer quickly and
superficially — MATH500 median response length rises 445 → 503 tokens between
the training script's baseline and final eval passes. On the hard benchmarks
both models already run near the 2048-token cap, so there is no "quick guess"
behavior to slow down from and throughput converges (18.7 vs 19.4, 23.6 vs
23.1, 35.7 vs 33.3 — all within noise).

## Results — Experiment 2 (7B)

**Experiment 1's result does not replicate at 7B.** ES still buys a real
pass@1 gain on every benchmark, but the gain shrinks monotonically with k, the
curves cross on 2 of 4 benchmarks, and on AIME24 the *base* model finishes far
ahead:

| Benchmark | pass@1 base → ES | pass@max base → ES | narrow / gain / **net** |
|---|---|---|---|
| AIME24 (n=30, k≤512) | 7.14% → 7.98% | 76.67% → 66.67% | 13.3% / 3.3% / **−10.0** |
| MATH500 (n=500, k≤128) | 61.15% → 67.54% | 95.40% → 95.80% | 1.0% / 1.4% / **+0.4** |
| Minerva (n=272, k≤128) | 23.14% → 26.99% | 65.81% → 65.81% | 2.9% / 2.9% / **0.0** |
| OlympiadBench (n=675, k≤128) | 27.95% → 32.39% | 74.67% → 76.30% | 3.4% / 5.0% / **+1.6** |

<table>
<tr>
<td><img src="results/7b-sigma001-iter50/aime24_passk.png" width="390" alt="AIME24 pass@k (7B)"></td>
<td><img src="results/7b-sigma001-iter50/math500_passk.png" width="390" alt="MATH500 pass@k (7B)"></td>
</tr>
<tr>
<td><img src="results/7b-sigma001-iter50/minerva_math_passk.png" width="390" alt="Minerva pass@k (7B)"></td>
<td><img src="results/7b-sigma001-iter50/olympiadbench_passk.png" width="390" alt="OlympiadBench pass@k (7B)"></td>
</tr>
</table>

The shape is clearest as ES − base in points, per k:

| k | AIME24 | MATH500 | Minerva | OlympiadBench |
|---|---|---|---|---|
| 1 | +0.85 | +6.39 | +3.85 | +4.44 |
| 2 | +1.03 | +3.99 | +3.39 | +3.95 |
| 4 | +1.27 | +2.32 | +1.97 | +3.38 |
| 8 | +1.26 | +1.48 | +0.53 | +2.88 |
| 16 | +0.90 | +0.97 | −0.32 | +2.32 |
| 32 | +0.69 | +0.45 | −0.55 | +2.01 |
| 64 | +0.01 | +0.12 | −0.29 | +1.91 |
| 128 | −1.44 | +0.40 | ±0.00 | +1.63 |
| 256 | −4.17 | — | — | — |
| 512 | **−10.00** | — | — | — |

Per benchmark:

- **AIME24**: the clean textbook case of the tradeoff — ES ahead through
  k≈32, tied at k=64, then falling behind, ending 10 points down at pass@512
  (76.7% base vs 66.7% ES). In question terms ES loses 4 of the 23 questions
  base can solve at k=512 and gains 1. n=30, so this is the noisiest curve of
  the four, but it is also the largest effect, and it is corroborated by the
  training suite's own AIME eval falling 16.7% → 6.7% over the run.
- **MATH500**: no crossover, but the gap collapses from +6.4 at k=1 to +0.4 at
  k=128. Both models are near-saturated by then (95.4% / 95.8%) — there is
  almost no ceiling left to move.
- **Minerva**: crosses at k≈16 and stays marginally below base through k=64,
  converging to an exact tie at k=128 (65.81% both). The dips are ≤0.6 points
  — read this as "flat ceiling", not real narrowing.
- **OlympiadBench**: the only benchmark that still looks like Experiment 1 —
  ES above base at every k, with the gap shrinking (+4.4 → +1.6) but never
  closing. Also the largest question set (675) and the only one with a clear
  net gain.

**Caveat that cuts both ways — headroom.** The 7B base is far stronger than
1.5B (MATH500 pass@128: 95.4% vs 78.6%; OlympiadBench 74.7% vs 44.7%), so
there is much less ceiling available to expand into, and a fixed 50-iteration
ES budget is a proportionally smaller intervention on a 7B model. Some of the
shrinking net gain is that, not necessarily "ES stops working at scale". What
this cannot explain is AIME24, where the ceiling *dropped* by 10 points — that
is capability loss, not saturation.

### The two experiments together

Net question-coverage change (gain − narrow), same ES recipe at both scales:

| Benchmark | 1.5B | 7B |
|---|---|---|
| AIME24 | **+13.3** | **−10.0** |
| MATH500 | **+10.6** | **+0.4** |
| Minerva | **+3.3** | **0.0** |
| OlympiadBench | **+9.4** | **+1.6** |

At 1.5B, ES looked like genuine capacity expansion. At 7B, the same recipe
looks like an almost pure sampling-efficiency gain — pass@1 up on all four
benchmarks, ceiling flat on three and down sharply on the fourth. That is the
profile the source RLVR paper attributes to gradient-based RLVR, which
weakens the project's original framing: "gradient-free ES avoids the
narrowing" does not hold as a scale-independent claim on this evidence.

Two honest limits on that conclusion: the 7B base has far less headroom (see
the caveat above), and there is still no compute-matched RL arm at either
scale — so this says something about ES across scale, and still nothing
rigorous about ES *versus* RLVR.

## Additional runs: perturbation scale and run length

Experiments 1 and 2 fix σ = 0.001 and 50 iterations. Five further 7B runs probe
the two obvious knobs around that point — is σ = 0.001 near-optimal, and does
running longer help? Both answers are no, in opposite directions. These arms
were scored on GPQA but **not** put through the pass@k suite (~3h35m of
generation each), so they inform hyperparameter choice, not the capacity claim.

Every run below is Qwen2.5-7B, population 32, `max_tokens=2048`, seed 42, on
the same 8x RTX 4090 box. "Reward" is the population mean of the binary math
reward; "Σ iter wall" sums the per-iteration times and so excludes startup.

| Run | σ | α | Batch | Iterations | Reward: start → end (peak) | Σ iter wall | Outcome |
|---|---|---|---|---|---|---|---|
| `7b-sigma001-iter50` (Exp 2) | 0.001 | auto, σ/2 | 256 | 1–51 | 0.457 → 0.654 (0.702 @ i25) | 4.34h | **the reference run** |
| `7b-sigma001-iter100` | 0.001 | auto, σ/2 | 256 | 52–101 | 0.664 → 0.645 (0.737 @ i75) | 4.20h | no further gain |
| `7b-sigma0025-iter50` | 0.0025 | σ/4 | 256 | 1–51 | 0.355 → 0.533 (0.649 @ i34) | 4.79h | worse than σ=0.001 |
| `7b-sigma0025-b64-aborted` | 0.0025 | σ/4 | 64 | 1–24 | 0.423 → 0.383 (0.511 @ i22) | 1.31h | stopped; relaunched at batch 256 |
| `7b-sigma005-b64-aborted` | 0.005 | σ/4 | 64 | 1–2 | 0.001 → 0.000 | 0.12h | **collapsed** |
| `7b-sigma01-b64-aborted` | 0.01 | σ/4 | 64 | 1–3 | 0.000 → 0.000 | 0.18h | **collapsed** |

### σ: 0.001 is at or near the ceiling, and the cliff above it is sharp

σ = 0.005 and σ = 0.01 both drove the population mean reward to **exactly zero
within two iterations** and were killed — every one of the 32 perturbed models
failed every problem in the batch. That is not slow degradation, it is the
perturbation destroying the model outright, and it also means ES gets no
gradient signal at all: with all-zero fitness there is nothing for rank shaping
to order, so the run could not have recovered on its own.

σ = 0.0025 stays alive but is clearly worse: it *starts* at 0.355 against
σ = 0.001's 0.457 (the perturbation is already costing accuracy at iteration 1)
and finishes at 0.533 against 0.654, never catching up. Its GPQA is also lower
(30.3% vs 31.8%). Note this run changed α as well as σ (σ/4 rather than the
auto σ/2), so σ and step size aren't cleanly separated here — but given the
total collapse just one step further out, the direction isn't in doubt.

So the usable range for full-rank ES on Qwen2.5-7B is narrow, and 0.001 sits
close to its upper edge. Nothing here tests *below* 0.001.

### Run length: iterations 51–100 bought nothing

The continuation resumes Experiment 2's checkpoint and runs 50 more iterations
for another 4.2 hours. Every measure is flat or slightly down:

| Measure | After 50 iters | After 100 iters | Δ |
|---|---|---|---|
| Training reward (population mean) | 0.664 | 0.645 | −0.019 |
| In-loop AIME | 6.67% | 6.67% | ±0.00 |
| In-loop AMC | 34.94% | 37.35% | +2.41 |
| In-loop MATH500 | 73.00% | 73.40% | +0.40 |
| In-loop Minerva | 37.87% | 37.87% | ±0.00 |
| In-loop OlympiadBench | 36.15% | 36.00% | −0.15 |
| GPQA-diamond | 31.8% | 28.8% | **−3.03** (4 gained / 10 lost, p = 0.18) |

Every in-loop move sits inside the ±3-point noise band measured
[above](#the-trainers-own-eval-suite-and-its-noise-floor), so the only reading
those rows support is "no change". Training reward peaked at 0.737 around
iteration 75 and drifted back down — the run is bouncing around a plateau it
had already reached by iteration ~25 of the *first* 50.

**This partly answers the headroom confound.** The open question after
Experiment 2 was whether 7B's weak result is "less headroom" (fixable with more
iterations) or "ES scales worse" (not). More iterations was the cheap test, and
it moved nothing — so the 7B result is not simply an under-trained 1.5B result.
It does not settle it either: the reward plateau means the *search* stalled,
which is what you'd expect if the population is too small for the parameter
count rather than if the model is out of headroom. Population, not iterations,
is the knob that hasn't been tried.

## Out-of-domain check: GPQA-diamond

All post-training here is math-only, so the obvious question is what it costs
elsewhere. GPQA-diamond zero-shot (198 graduate-level science questions,
4-choice, scored by log-likelihood — **not** pass@k; a different probe than
everything above), run across all 8 arms with `lm_eval`:

| Arm | acc | vs. its base | McNemar (paired, exact) |
|---|---|---|---|
| 1.5B base | 26.3% (52/198) | — | — |
| 1.5B ES | **34.8% (69/198)** | **+8.6** | 27 gained / 10 lost, **p = 0.008** |
| 1.5B RL (SimpleRL-Zoo) | 27.3% (54/198) | +1.0 | 5 gained / 3 lost, p = 0.73 |
| 7B base | 32.8% (65/198) | — | — |
| 7B ES (σ=0.001, iter50) | 31.8% (63/198) | −1.0 | 9 gained / 11 lost, p = 0.82 |
| 7B ES (σ=0.0025, iter50) | 30.3% (60/198) | −2.5 | 12 gained / 17 lost, p = 0.46 |
| 7B ES (σ=0.001, iter100) | 28.8% (57/198) | −4.0 | 6 gained / 14 lost, p = 0.12 |
| 7B RL (SimpleRL-Zoo) | 32.8% (65/198) | ±0.0 | 2 gained / 2 lost, p = 1.00 |

**At 1.5B, a large gain in the wrong direction from what "forgetting" predicts.**
**Math-only ES training improves out-of-domain science QA by 8.6 points** — the
only change in the table that survives a paired test. Note where it starts
from: 4-choice chance is 25%, and 1.5B base (26.3%) and 1.5B RL (27.3%) are
both indistinguishable from guessing, so ES is the only 1.5B arm doing better
than chance at all. Whether that is real science knowledge or just
better-calibrated option scoring on a model that could previously do neither,
this run can't say.

**At 7B, a consistent downward drift that no single test can confirm.** All
three 7B ES arms land below their base (−1.0, −2.5, −4.0), and the ordering
tracks how much the weights were moved: further from the base weights (more
iterations, or a larger σ) means lower GPQA. No individual arm's drop is
significant at n=198 — the best is p = 0.12 — and the ES arms are not
independent of each other (iter100 *is* iter50, trained further), so this is
one weak signal, not three. The paired iter50 → iter100 comparison is the
cleanest version of it: −3.0 points, 4 gained / 10 lost, p = 0.18.

Read together: at 1.5B the ES run was a large intervention on a weak model and
moved things in *both* domains; at 7B it was a small intervention that left
in-domain pass@k roughly flat and may be slowly costing out-of-domain accuracy
as it runs longer. What the table does rule out is a *large* out-of-domain
collapse at 7B — so Experiment 2's AIME24 narrowing isn't a symptom of general
degradation. What it no longer supports is a flat "no forgetting anywhere".

Driver: `scripts/run_gpqa_sweep.sh` (single GPU, vLLM backend, ~1.5 min/model,
~9 min for the sweep), reduced to committed CSVs by `scripts/analyze_gpqa.py`.
Per-arm scores, the full per-question correctness matrix, and every McNemar
pair: [`results/gpqa/`](results/gpqa/).

## Training dynamics

Experiment 2 ran with `--logging none`, so there is no W&B run behind it — its
per-iteration stats come from the trainer's stdout, which records reward only.
Both runs are therefore reported the same way here. Reward across the
population of 32, at four points in each 51-iteration run:

| | Iter 1 | Iter 10 | Iter 25 | Iter 51 |
|---|---|---|---|---|
| Mean reward — 1.5B | 0.014 | 0.083 | 0.400 | 0.428 |
| Mean reward — 7B | 0.457 | 0.549 | 0.702 | 0.654 |
| Std — 1.5B | 0.009 | 0.045 | 0.037 | 0.020 |
| Std — 7B | 0.050 | 0.044 | 0.018 | 0.017 |

Both follow the same shape — reward climbs, plateaus around iteration ~25, std
decays as the population converges — with two differences worth noting. The 7B
run *starts* at 0.457, above where the 1.5B run *finishes* (0.428), so nearly
all of the 1.5B run's headroom on the training reward simply isn't there; total
improvement is +0.20 (7B) vs +0.41 (1.5B). And 7B's std decays monotonically
instead of peaking early around iteration ~10-15 — the population never gets
the broad early exploration phase the 1.5B run had. Both are noisy after the
plateau (7B bounces 0.61-0.70), i.e. the last ~25 iterations buy little.

W&B additionally captured response length for Experiment 1, with no 7B
counterpart to compare against: the population mean *falls* the whole run,
~1708 tokens at iteration 1 to ~787 by iteration 51, so the 1.5B model learns
to be **more concise while getting more correct** — not to think longer.
Full per-iteration series: `results/1.5b-sigma001-iter50/wandb_curves.csv`
(`scripts/plot_training_curves.py` will plot it, or regenerate from
`--wandb-run chunhinma00-personal/es-finetuning/it2de910`).

Per-iteration reward for every run is in `results/<run>/training_curves.csv`,
extracted from the trainer's stdout by `scripts/parse_training_log.py`.

### The trainer's own eval suite, and its noise floor

The training script also runs its own eval suite (single-sample pass@1, its own
prompt/sampling settings — **not** comparable to the pass@k harness numbers
above). For **Experiment 2 (7B)**, baseline vs. final:

| Task | Baseline | Iter 50 |
|---|---|---|
| AIME | 16.7% | **6.7%** |
| AMC | 34.9% | 41.0% |
| MATH500 | 46.8% | 71.4% |
| Minerva | 21.0% | 34.6% |
| OlympiadBench | 28.9% | 37.2% |

Large gains everywhere except AIME, which halves — the same direction as the
AIME24 narrowing in the pass@k results, from a different harness.

**This suite is not deterministic, which is worth knowing before leaning on it.**
The additional runs happen to supply two free repeated measurements — every run
evaluates its starting weights before training anything, so the pristine 7B base
was scored twice (Experiment 2 and the aborted σ=0.01 run), and the Experiment 2
checkpoint was scored twice (as Experiment 2's final eval and as the
100-iteration continuation's baseline). Same weights, same command, same seed:

| Task | Base, run A → B | Δ | ES iter50, run A → B | Δ |
|---|---|---|---|---|
| AIME (n=30) | 16.67% → 16.67% | ±0.00 | 6.67% → 6.67% | ±0.00 |
| AMC (n=83) | 34.94% → 34.94% | ±0.00 | 40.96% → 34.94% | **−6.02** (5 q) |
| MATH500 (n=500) | 46.80% → 46.60% | −0.20 (1 q) | 71.40% → 73.00% | +1.60 (8 q) |
| Minerva (n=272) | 20.96% → 19.85% | −1.10 (3 q) | 34.56% → 37.87% | +3.31 (9 q) |
| OlympiadBench (n=675) | 28.89% → 30.37% | +1.48 (10 q) | 37.19% → 36.15% | −1.04 (7 q) |

Between 1 and 10 questions flip on the larger benchmarks in each pair, so the
harness genuinely is nondeterministic — presumably vLLM batching across the 8
co-located engines. The worst single cell is AMC's 6 points, though that same
cell was stable in the other pair, so treat ~±3 points as the working noise
band on the mid-size benchmarks rather than 6.

**AIME reproduced exactly in both pairs**, which is the relevant fact for the
16.7% → 6.7% drop: nothing here suggests the number wanders on its own. That
makes it a real corroborating signal for the AIME24 pass@k narrowing rather
than an artifact. It is still only 3 questions out of 30 from a single-sample
harness, so the load-bearing evidence remains the n_sampling=512 pass@k run,
where the same effect is measured over 15,360 completions.

## Model

The two headline checkpoints are published, in HF format, converted from the raw
`es-at-scale` checkpoints with `scripts/convert_to_hf.py --verify`:

| Run | Checkpoint |
|---|---|
| Experiment 1 — 1.5B, σ=0.001, iter50 | [zocrate/Qwen2.5-1.5B-ES-math](https://huggingface.co/zocrate/Qwen2.5-1.5B-ES-math) |
| Experiment 2 — 7B, σ=0.001, iter50 | [zocrate/Qwen2.5-7B-ES-math](https://huggingface.co/zocrate/Qwen2.5-7B-ES-math) |
| 7B continuation, σ=0.001, iter100 | [zocrate/Qwen2.5-7B-ES-math-iter100](https://huggingface.co/zocrate/Qwen2.5-7B-ES-math-iter100) |

The iter100 checkpoint is the Experiment 2 run continued for 50 more
iterations. It was not put through the pass@k suite; in-loop eval and GPQA
are strictly no-better / slightly worse than iter50. The remaining
unpublished additional-run checkpoint is 7B σ=0.0025 iter50. Full numbers
for both are in `results/`.

## Later

- **Chase the 1.5B GPQA gain.** +8.6 points out-of-domain from math-only ES
  training (p = 0.008) is the most interesting unexplained result in the repo,
  and it rests on 198 questions against a base model at chance. Worth a second
  seed and a second out-of-domain benchmark (MMLU-STEM, ARC-Challenge) before
  claiming transfer.
- **Is AIME24's 7B narrowing real or n=30 noise?** It is the strongest result
  in Experiment 2 and rests on 30 questions (4 lost, 1 gained). Worth a second
  seed, or AIME25/AMC as an independent hard-benchmark check, before leaning
  on it. The trainer's in-loop AIME eval does survive a
  [reproducibility check](#the-trainers-own-eval-suite-and-its-noise-floor)
  (identical across two repeat measurements), so it corroborates — but it is
  the same 30 questions, not an independent question set.
- **Population, not iterations.** The 100-iteration run
  ([above](#run-length-iterations-51100-bought-nothing)) killed the "just train
  longer" version of the headroom confound: 50 more iterations moved nothing
  and the reward plateau held. The untested knob is population size. Full-rank
  ES estimates a descent direction in a D-dimensional space from `population`
  samples, and D grows ~4.9x from 1.5B to 7B while population stayed at 32 —
  so each 7B update is drawn from a proportionally much noisier estimate. A 7B
  run at population 128 or 256, iterations held at 50, is the experiment that
  separates "ES scales worse" from "population 32 is too small at 7B". Cost is
  linear in population, so ~4x/~8x the 4.3h.
- **Does the 7B GPQA drift compound?** All three 7B ES arms sit below base and
  the gap grows with distance from the base weights, but no single arm reaches
  significance. A 200-iteration run would say whether this is a real monotone
  cost or three draws of noise — it is cheap to check as a side effect of any
  longer run.
- No RL arm in the 7B pass@k suite (`hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo` is
  downloaded and is in the GPQA sweep, just not the math suite — another ~2h
  of generation).
- A compute-matched RL baseline (capped at similar wall-clock/FLOP budget to
  the ES run, rather than comparing against an already fully-trained public
  checkpoint) would make the ES-vs-RL comparison fair — currently open at both
  scales.
- A fourth arm testing EGGROLL (Sarkar et al., arXiv:2511.16652 — rank-r
  LoRA-factorized ES, as opposed to `es-at-scale`'s full-rank ES).
