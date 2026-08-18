# ES-capacity

Does Evolution-Strategies (ES) post-training preserve a base model's pass@k
ceiling better than gradient-based RLVR (GRPO)? RLVR reliably raises pass@1
but tends to narrow the pass@k ceiling relative to the base model — this
project tests whether ES-based fine-tuning avoids that narrowing, on the same task/reward/data.

Run so far at two scales (1.5B, 7B) on Qwen2.5 base models. There are early hints that ES preserve the ceiling and avoid narrowing. More experiments are required to verify this claim. 


We used the same set of hyperparameters for ES fine-tuning on 1.5B and 7B, and compared them with RL models published with SimpleRL.
- 1.5B ES and RL expand
question coverage on all 4 benchmarks with no pass@k crossover
- 7B ES has higher pass@1 on all 4 benchmarks, crosses base on AIME24; 7B RL has higher pass@1 on all 4 benchmarks, crosses base on all 4 benchmarks.

Note that the RL arm throughout is a published SimpleRL-Zoo checkpoint, with much larger compute budget, and was trained with about **240 H100 hours**. On the other hand, the ES model was trained with about **36 RTX4090 hours**. 

The RL model saw
**12.01 epochs of the shared 8,523-problem training set against this ES run's
1.50** ([arithmetic](#results--experiment-1-qwen25-15b)). The ES models visit **8x fewer times over the same dataset**.

Training and evaluation run from forked, patched copies of the original
papers' code.

## Code

| Purpose | Repo | Branch | Notes |
|---|---|---|---|
| ES training | [Jaysen-Ma/es-at-scale](https://github.com/Jaysen-Ma/es-at-scale) (fork of [VsonicV/es-at-scale](https://github.com/VsonicV/es-at-scale), arXiv:2509.24372) | `fix/multi-engine-colocation` | Fixes needed to run on a single-node 8x RTX 4090 instance on Vast. |
| pass@k generation, grading, plotting | [Jaysen-Ma/limit-of-RLVR](https://github.com/Jaysen-Ma/limit-of-RLVR) (fork of [LeapLabTHU/limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR), arXiv:2504.13837) | `fix/math-equal-timeout-bypass` | Fixes a grading-worker hang. |

## Experiment 1: Qwen2.5-1.5B base vs. ES-at-scale-trained (done)

| Param | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B` (Base) |
| Task | math |
| sigma | 0.001 |
| alpha (lr) | auto (`sigma/2` = 0.0005) |
| Population size | 32 |
| Iterations | 50 |
| Train dataset | `math_lvl3to5_8k` — the same 8,523 problems, in the same order, as SimpleRL-Zoo's `simplelr_qwen_level3to5` split|
| Evaluation dataset | `datasets/evaluation_suite/math` — omit it and it defaults to the `countdown` task's, which has no `problem` field and crashes the trainer |
| Batch size / mini-batch size | 256 / 256 |
| Max tokens | 2048 |
| vLLM engines | 8 (one per GPU) |
| GPUs | 0-7 (8x RTX 4090 48GB) |
| Training wall-clock | 3h 22m 35s (including 2 evals) |


### Evaluation wall-clock

Convert the raw `es-at-scale` checkpoint to HF format first.

Per-benchmark generation time (base, ES-trained, RL), 8x RTX 4090 48GB,
sharded across all 8 GPUs, `n_sampling`/model/benchmark:

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
`top_p=0.95`, `max_tokens=2048`, `qwen-boxed` template, `seed=1`). `n_sampling`
(= max k) is 512 for AIME24, 128 for the rest; each benchmark's full data
range gets its own pass@k curve.

- **Base**: `Qwen/Qwen2.5-1.5B`
- **ES**: [this project's Experiment 1 checkpoint](https://huggingface.co/zocrate/Qwen2.5-1.5B-ES-math)
- **RL**: [hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo),
  a published GRPO-trained model, same base and (matched) training data

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

Solvable/unsolvable breakdown ("solvable" = at least 1 of `n_sampling`
completions correct):

| Benchmark | ES: narrow / gain / **net** | RL: narrow / gain / **net** |
|---|---|---|
| AIME24 | 0.0% / 13.3% / **+13.3** | 3.3% / 30.0% / **+26.7** |
| MATH500 | 1.4% / 12.0% / **+10.6** | 1.2% / 12.6% / **+11.4** |
| Minerva | 7.0% / 10.3% / **+3.3** | 5.5% / 12.5% / **+7.0** |
| OlympiadBench | 3.9% / 13.3% / **+9.4** | 3.4% / 12.9% / **+9.5** |

**ES vs. RL** 

**Not a compute-matched comparison — and the mismatch is different on each
axis.** Both arms train on the same **8,523 problems** (level3to5), but they
spend their budget on different things:

| | Prompts/step | Steps | ×per prompt | Prompt-exposures | **Epochs** | Generations | Token cap |
|---|---|---|---|---|---|---|---|
| **ES** (this run) | 256 | 50 | 32 population | 12,800 | **1.50** | 409,600 | 2,048 |
| **RL** (SimpleRL-Zoo 1.5B) | 1,024 | ~100 | 8 rollouts | 102,400 | **12.01** | 819,200 | 8,192 |
| **ratio (RL / ES)** | 4x | 2x | 0.25x | **8.00x** | **8.00x** | **2.00x** | **4x** |

The two ratios that matter come apart: RL sees **8x more data** but runs only
**2x more generations**. That is the whole structural difference between the
methods. RL spends its generation budget on **data breadth**, ES spends its
budget on **parameter breadth**. 

**On "rollouts" vs "population"**: 

Both multiply the
per-step generation count. GRPO rollout
is a sample from one policy used to estimate an advantage for one prompt, and
the actual descent direction comes from backprop. An ES population member is a
different point in parameter space, and the population *is* the entire gradient
estimator.

### Reading any ES-vs-RL curve in this repo

Three differences apply to every ES-vs-RL comparison here, at both scales. They
are not fatal — the curves are still worth showing — but each one has a
direction, so read the comparison with them in mind rather than as a clean
head-to-head.

| Difference | ES | RL | Which way it cuts |
|---|---|---|---|
| Training token cap | 2,048 | 8,192 | Favours RL on long-form problems. Our eval caps generation at 2,048, so RL's extra training-time length can't be *scored* here, but it may still have shaped the policy. |
| Data budget | 1.50 epochs | 12.01 epochs | Favours RL, heavily. |
| Generations | 409,600 | 819,200 | Favours RL, 2x. |
| Compute budget | 8 * RTX4090 | 2 nodes of 8 * H100 | Maybe I will own a giant cluster of B300s soon. |

## Experiment 2: Qwen2.5-7B base vs. ES-at-scale-trained (done)

Same task, same reward, same data, same ES hyperparameters as Experiment 1 —
**only the base model changes** (1.5B → 7B). The point is to test whether
Experiment 1's clean capacity-expansion result is a property of ES or a
property of a small, weak base model with a lot of headroom.

| Param | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B` (base, not Instruct) |
| Training wall-clock | 4h 35m 33s |

### Evaluation wall-clock

| Benchmark | Gens/model | Base | ES-trained | Base gen/s | ES gen/s |
|---|---|---|---|---|---|
| AIME24 (n=512) | 15,360 | 13m41s | 13m12s | 18.7 | 19.4 |
| MATH500 (n=128) | 64,000 | 11m19s | 26m07s | 94.3 | 40.8 |
| Minerva (n=128) | 34,816 | 24m33s | 25m07s | 23.6 | 23.1 |
| OlympiadBench (n=128) | 86,400 | 40m21s | 43m17s | 35.7 | 33.3 |
| **Total generation** | | **1h30m** | **1h48m** | | |

## Results — Experiment 2 (7B)

Three models compared, same as Experiment 1: **Base** (`Qwen/Qwen2.5-7B`),
**ES** (this project's Experiment 2 checkpoint), and **RL**
([hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo),
a published GRPO-trained model on the same base). The RL arm's generations
were added in a later pass, after base/ES; see the caveat under [Evaluation
wall-clock](#evaluation-wall-clock-1) above.

**Experiment 1's result does not replicate at 7B — for ES.** ES still buys a
real pass@1 gain on every benchmark, but the gain shrinks monotonically with
k, the curves cross on 2 of 4 benchmarks, and on AIME24 the *base* model
finishes far ahead. **RL looks the opposite of its 1.5B self**: it has by far
the largest pass@1 lift of the three arms on every benchmark, and by far the
largest ceiling loss — it narrows on all 4 benchmarks, more than ES on every
one of them:

| Benchmark | pass@1 base→ES→RL | pass@max base→ES→RL | ES narrow/gain/**net** | RL narrow/gain/**net** |
|---|---|---|---|---|
| AIME24 (n=30, k≤512) | 7.14%→7.98%→15.39% | 76.67%→66.67%→56.67% | 13.3/3.3/**−10.0** | 23.3/3.3/**−20.0** |
| MATH500 (n=500, k≤128) | 61.15%→67.54%→76.04% | 95.40%→95.80%→93.80% | 1.0/1.4/**+0.4** | 3.0/1.4/**−1.6** |
| Minerva (n=272, k≤128) | 23.14%→26.99%→35.38% | 65.81%→65.81%→62.13% | 2.9/2.9/**0.0** | 5.1/1.5/**−3.6** |
| OlympiadBench (n=675, k≤128) | 27.95%→32.39%→38.59% | 74.67%→76.30%→73.78% | 3.4/5.0/**+1.6** | 5.5/4.6/**−0.9** |

<table>
<tr>
<td><img src="results/7b-sigma001-iter50/aime24_threeway_passk.png" width="390" alt="AIME24 pass@k (7B)"></td>
<td><img src="results/7b-sigma001-iter50/math500_threeway_passk.png" width="390" alt="MATH500 pass@k (7B)"></td>
</tr>
<tr>
<td><img src="results/7b-sigma001-iter50/minerva_math_threeway_passk.png" width="390" alt="Minerva pass@k (7B)"></td>
<td><img src="results/7b-sigma001-iter50/olympiadbench_threeway_passk.png" width="390" alt="OlympiadBench pass@k (7B)"></td>
</tr>
</table>

### The two experiments together

Net question-coverage change (gain − narrow), both methods at both scales:

| Benchmark | 1.5B ES | 1.5B RL | 7B ES | 7B RL |
|---|---|---|---|---|
| AIME24 | +13.3 | +26.7 | **−10.0** | **−20.0** |
| MATH500 | +10.6 | +11.4 | **+0.4** | **−1.6** |
| Minerva | +3.3 | +7.0 | **0.0** | **−3.6** |
| OlympiadBench | +9.4 | +9.5 | **+1.6** | **−0.9** |

At 1.5B, ES looked like genuine capacity expansion. At 7B, the same recipe
looks like an almost pure sampling-efficiency gain — pass@1 up on all four
benchmarks, ceiling flat on three and down sharply on the fourth. That is the
profile the source RLVR paper attributes to gradient-based RLVR, which
weakens the project's original framing: "gradient-free ES avoids the
narrowing" does not hold as a scale-independent claim on this evidence.

**But RL's own narrowing is the more consistent pattern of the two.** At
1.5B, RL narrowed *less* than ES on 3 of 4 benchmarks (see [Results —
Experiment 1](#results--experiment-1-qwen25-15b)); at 7B that reverses
completely — RL narrows on all 4 benchmarks, and by a wider margin than ES on
every one. ES's narrowing-resistance doesn't hold across scale, but neither
does RL's *lack* of it — RL looks bad at both scales, worse at 7B, while ES
is the arm that's clean at one scale and merely flat-to-down at the other.
Compared to its own base, ES's story is scale-dependent; compared to RL
directly, ES comes out ahead at both scales on this measure.

Two honest limits on that conclusion: the 7B base has far less headroom (see
the caveat above), and there is still no compute-matched RL arm at either
scale — so the ES-vs-base finding says something about ES across scale, but
the ES-vs-RL comparison above is still not a rigorous "ES vs RLVR" claim,
just a same-checkpoints comparison at both scales.

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

## Training dynamics

Per-iteration reward for every run is in `results/<run>/training_curves.csv`

## Model

Published checkpoints, in HF format, converted from the raw `es-at-scale`
checkpoints. `es-at-scale` saves a raw state_dict straight from vLLM's
internal Qwen2 model, which fuses attention QKV into one `qkv_proj` tensor
and MLP gate+up into one `gate_up_proj` tensor — HF's `AutoModelForCausalLM`
expects those split back into `q_proj`/`k_proj`/`v_proj` and
`gate_proj`/`up_proj` (every other parameter name matches already). The
converter that did this split was Qwen2.5-specific and isn't part of this
repo — write your own for whatever architecture you're training if you need
this step

| Run | Checkpoint |
|---|---|
| Experiment 1 — 1.5B, σ=0.001, iter50 | [zocrate/Qwen2.5-1.5B-ES-math](https://huggingface.co/zocrate/Qwen2.5-1.5B-ES-math) |
| Experiment 2 — 7B, σ=0.001, iter50 | [zocrate/Qwen2.5-7B-ES-math](https://huggingface.co/zocrate/Qwen2.5-7B-ES-math) |
| 7B continuation, σ=0.001, iter100 | [zocrate/Qwen2.5-7B-ES-math-iter100](https://huggingface.co/zocrate/Qwen2.5-7B-ES-math-iter100) |

The iter100 checkpoint is the Experiment 2 run continued for 50 more
iterations. It was not put through the pass@k suite; 

## Planned: ''matched-budget'' ES vs. RL

**Not yet run.** The single biggest weakness in this repo is that every
ES-vs-RL number compares an ES run to an already-fully-trained public
checkpoint.

**Move GRPO instead.** Retrain the RL arm at **batch 256, 32 samples per
prompt** — ES's exact geometry:

| | Steps | Prompts/step | ×per prompt | Exposures | Epochs | Generations |
|---|---|---|---|---|---|---|
| ES iter50 | 50 | 1024 | 32 population | 51,200 | 6 | 1,638,400 |
| **GRPO reshaped** | 50 | 1024 | 32 rollouts | 51,200 | 6 | 1,638,400 |
| ES iter100 | 100 | 1024 | 32 population | 102,400 | 12 | 3,276,800 |
| **GRPO reshaped** | 100 | 1024 | 32 rollouts | 102,400 | 12 | 3,276,800 |

### Does reshaping break GRPO? Two effects, opposite signs

### The protocol

1. **Train our own GRPO arm** at batch 1024 / 32 rollouts, rather than using the
   public checkpoint — the only way to control the budget. Also set
   `max_response_length=2048` to match ES and the eval, removing the token-cap
   confound in the same stroke.
2. **Run them with the same steps**, at 50 and/or 100steps.
3. **Record wall-clock and total generated tokens alongside generations.**
   Generation parity is not FLOP parity: GRPO adds a backward pass ES does not
   have. Report all three and let the reader choose a denominator.
4. **Run the existing pass@k suite**, so we have a more fair competition between ES and RLVR.

## Planned: full-MMLU forgetting measurement

**Not yet run.** All post-training here is math-only, so the natural question
is what it costs elsewhere.

*Catastrophic forgetting* would be a significant drop vs. base
concentrated in the non-STEM categories (humanities, social sciences) while
math-adjacent STEM holds or rises — that is the signature of a math-only
objective overwriting unrelated capability. 

## Later
- A compute-matched RL baseline (capped at similar wall-clock/FLOP budget to
  the ES run, rather than comparing against an already fully-trained public
  checkpoint) would make the ES-vs-RL comparison fair — currently open at both
  scales.
- A fourth arm testing EGGROLL (Sarkar et al., arXiv:2511.16652 — rank-r
  LoRA-factorized ES, as opposed to `es-at-scale`'s full-rank ES).
