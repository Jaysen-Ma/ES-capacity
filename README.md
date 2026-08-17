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
documents, now showing up in gradient-free ES. Out-of-domain, an earlier
GPQA-diamond sweep found a large unexplained *gain* at 1.5B and a consistent
but individually-insignificant downward drift across all three 7B ES arms —
paused pending a redo ([below](#out-of-domain-check-gpqa-diamond)).

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

Evaluate — one call per arm, each running all 4 benchmarks through identical
settings, sharded across every GPU (`n_sampling` = 512 for AIME24, 128 for
the rest), then one call to analyze every benchmark at once. `run_eval.sh`
needs `MATH_EVAL_DIR` pointing at a `limit-of-RLVR` checkout, and if a model
argument names a variable from `config.sh` it resolves to that variable's
path — each model gets its own explicit entry, there's no shared "models
root" assumed (copy `config.example.sh` to `config.sh` and fill in both, or
export the variables yourself — see the script's header):
```bash
cd ES-capacity
cp config.example.sh config.sh && vi config.sh   # once per environment
scripts/run_eval.sh QWEN25_1_5B_BASE base 1.5b-sigma001-iter50   # resolves $QWEN25_1_5B_BASE
scripts/run_eval.sh QWEN25_1_5B_ES trained 1.5b-sigma001-iter50
scripts/run_eval.sh QWEN25_1_5B_RL rl 1.5b-sigma001-iter50 [n_sampling_override]

scripts/analyze_passk.py --run-tag 1.5b-sigma001-iter50 \
  --label base --label trained --label rl --baseline base --plot
```
Convert the raw `es-at-scale` checkpoint to HF format first, if starting
from one — see [Model](#model) below. `analyze_passk.py` works with any 2+
labels (the `rl` arm is optional) and skips any benchmark missing for a requested label
rather than failing the whole run.

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
one shared wrapper so runs can't drift: `scripts/run_eval.sh`). `n_sampling`
(= max k) is 512 for AIME24, 128 for the rest; each benchmark's full data
range gets its own pass@k curve.

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

**Not a compute-matched comparison — and the mismatch is different on each
axis.** Both arms train on the same **8,523 problems** (SimpleRL-Zoo's
`simplelr_qwen_level3to5` split; verified identical, same order), but they
spend their budget on different things:

| | Prompts/step | Steps | ×per prompt | Prompt-exposures | **Epochs** | Generations | Token cap |
|---|---|---|---|---|---|---|---|
| **ES** (this run) | 256 | 50 | 32 population | 12,800 | **1.50** | 409,600 | 2,048 |
| **RL** (SimpleRL-Zoo 1.5B) | 1,024 | ~100 | 8 rollouts | 102,400 | **12.01** | 819,200 | 8,192 |
| **ratio (RL / ES)** | 4x | 2x | 0.25x | **8.00x** | **8.00x** | **2.00x** | **4x** |

The two ratios that matter come apart: RL sees **8x more data** but runs only
**2x more generations**. That is the whole structural difference between the
methods. RL spends its generation budget on *data breadth* — many prompts, 8
samples each, just enough to form a group-relative advantage. ES spends its
budget on *parameter breadth* — few prompts, but 32 perturbed copies of the
entire network evaluated on all of them. Same benchmark, same reward, very
different thing being estimated.

**On pairing "rollouts" with "population": they are the right analogy for
counting generations and the wrong one for statistics.** Both multiply the
per-step generation count, so the arithmetic above is sound. But a GRPO rollout
is a sample from one policy used to estimate an advantage for one prompt, and
the actual descent direction comes from backprop. An ES population member is a
different point in parameter space, and the population *is* the entire gradient
estimator — there is no backward pass at all. So ES trades an exact gradient
for a `population`-sample estimate of a D-dimensional one, which is why it can
look generation-hungry and FLOP-cheap at the same time. Don't read "32 > 8" as
ES getting more of the same resource.

Two caveats on the figures themselves. The step count is read off SimpleRL-Zoo's
per-model figures for **Qwen2.5-1.5B**, since the paper never states RL epochs
directly — it is not independently verified for their 7B run, so treat the 7B
version of this table as inheriting that assumption. And the paper describes the
split as "approximately 8,000 problems" (§3.2) — dividing by that round number
is where the commonly-quoted **12.8** epochs comes from; against the actual
8,523-row file it is 12.01, and the matching ES figure moves 1.60 → 1.50. Either
pairing gives the same 8.00x.

That reframes the ES-vs-RL rows above: RL matching or slightly beating ES here
is what an 8x larger data budget should buy. It doesn't rule out ES preserving
capacity better under matched compute — that comparison still hasn't been run,
and [what "matched" should even mean](#planned-matched-budget-es-vs-rl) is not
obvious. What holds regardless, at this scale: ES itself shows no crossover and
a clean net-positive result relative to the base model on all 4 benchmarks, off
1.50 epochs.

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
| Checkpoint shown | **iter50** at both scales | fully trained | The 7B iter100 checkpoint exists and doubles ES's generations to exact parity with RL, but was never put through the pass@k suite — so every pass@k curve here is the half-budget ES model. |

The last row is the one most likely to be misread. Where a curve says "ES", it
is the 50-iteration checkpoint, for both 1.5B and 7B.

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

Evaluate — same wrapper and same fixed settings as Experiment 1, with the 7B
base as its own `model_dir`:
```bash
cd ES-capacity
scripts/run_eval.sh /workspace/.hf_home/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 base 7b-sigma001-iter50
scripts/run_eval.sh /workspace/es-at-scale/experiments/qwen7b-math-run/hf-checkpoint-iter50 trained 7b-sigma001-iter50
```

### Evaluation wall-clock

Base then ES-trained, sequential per benchmark, same 8-GPU sharding. The RL
arm ([hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo))
was generated in a separate later pass via `scripts/run_eval.sh`; its
wall-clock wasn't captured separately and isn't in the table below.

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

Per-k numbers for all three arms: **[results/README.md](results/README.md)**.
The shape is clearest as ES − base / RL − base in points, per k:

| k | AIME24 ES/RL | MATH500 ES/RL | Minerva ES/RL | OlympiadBench ES/RL |
|---|---|---|---|---|
| 1 | +0.85 / **+8.26** | +6.39 / **+14.89** | +3.85 / **+12.24** | +4.44 / **+10.64** |
| 2 | +1.03 / +7.74 | +3.99 / +8.06 | +3.39 / +8.48 | +3.95 / +8.04 |
| 4 | +1.27 / +6.11 | +2.32 / +3.89 | +1.97 / +4.33 | +3.38 / +6.11 |
| 8 | +1.26 / +4.71 | +1.48 / +1.39 | +0.53 / +1.20 | +2.88 / +4.57 |
| 16 | +0.90 / +4.45 | +0.97 / −0.19 | −0.32 / −1.04 | +2.32 / +2.98 |
| 32 | +0.69 / +4.69 | +0.45 / −1.24 | −0.55 / −2.71 | +2.01 / +1.56 |
| 64 | +0.01 / +3.61 | +0.12 / −1.72 | −0.29 / −3.65 | +1.91 / +0.48 |
| 128 | −1.44 / **−0.48** | +0.40 / **−1.60** | ±0.00 / **−3.68** | +1.63 / **−0.89** |
| 256 | −4.17 / −8.18 | — | — | — |
| 512 | **−10.00 / −20.00** | — | — | — |

Per benchmark:

- **AIME24**: the clean textbook case of the ES/base tradeoff — ES ahead
  through k≈32, tied at k=64, then falling behind, ending 10 points down at
  pass@512 (76.7% base vs 66.7% ES). In question terms ES loses 4 of the 23
  questions base can solve at k=512 and gains 1. n=30, so this is the
  noisiest curve of the four, but it is also the largest effect, and it is
  corroborated by the training suite's own AIME eval falling 16.7% → 6.7%
  over the run. **RL is the extreme version of the same story**: best pass@1
  of the three (15.4%), briefly ties ES around k=64, then falls off a cliff —
  56.7% at pass@512, 20 points under base and 10 under ES.
- **MATH500**: ES has no crossover, but the gap collapses from +6.4 at k=1 to
  +0.4 at k=128 — both models near-saturated by then (95.4% / 95.8%), almost
  no ceiling left to move. **RL crosses below base around k≈16** despite a
  pass@1 nearly 15 points above base (76.0% vs 61.2%), finishing 1.6 points
  under base and 2.0 under ES at k=128.
- **Minerva**: ES crosses at k≈16 and stays marginally below base through
  k=64, converging to an exact tie at k=128 (65.81% both) — read this as
  "flat ceiling", not real narrowing. **RL is Minerva's worst case for
  ceiling loss**: crosses below base at k≈16 and finishes 3.7 points under
  it, the largest RL narrowing of the four benchmarks, despite having the
  benchmark's largest pass@1 lift (+12.2 over base).
- **OlympiadBench**: the only benchmark where ES still looks like
  Experiment 1 — above base at every k, gap shrinking (+4.4 → +1.6) but never
  closing, the largest question set (675) and ES's only clear net gain.
  **RL stays above base the longest of the four benchmarks too**, only
  dipping below at the very end (k=128: 73.78% vs base 74.67%, −0.89) — the
  smallest RL narrowing, mirroring it being ES's best case as well.

**Caveat that cuts both ways — headroom.** The 7B base is far stronger than
1.5B (MATH500 pass@128: 95.4% vs 78.6%; OlympiadBench 74.7% vs 44.7%), so
there is much less ceiling available to expand into, and a fixed 50-iteration
ES budget is a proportionally smaller intervention on a 7B model. Some of the
shrinking net gain is that, not necessarily "ES stops working at scale". What
this cannot explain is AIME24, where the ceiling *dropped* by 10 points — that
is capability loss, not saturation. The same headroom logic applies even more
to RL's pass@1 lift, which is not compute-matched here either (see [Reading
any ES-vs-RL curve in this repo](#reading-any-es-vs-rl-curve-in-this-repo));
what it doesn't explain is why RL's *ceiling loss* is uniformly larger than
ES's across all 4 benchmarks, not just its pass@1 gain.

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

Paused pending a redo. An earlier zero-shot sweep (198 graduate-level science
questions, 4-choice, scored by log-likelihood — **not** pass@k — `lm_eval`,
all 8 arms) found a large, statistically significant out-of-domain gain from
1.5B ES training (+8.6 points vs. its base, p = 0.008) and a smaller,
individually-insignificant downward drift across all three 7B ES arms
(−1.0 to −4.0 points, best p = 0.12) — opposite signs, on a 1.5B base model
sitting at chance (26.3% vs. 25% chance), which is exactly the profile that's
worth re-confirming with a second seed and a bigger out-of-domain benchmark
rather than trusting from one 198-question, single-seed sweep. The sweep
script and reduced CSVs have been removed pending that redo; see
[Later](#later).

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
(kept locally, not committed) or the W&B run itself,
`chunhinma00-personal/es-finetuning/it2de910`. The four `train_*.png` plots
derived from it are likewise local only.

Per-iteration reward for every run is in `results/<run>/training_curves.csv`,
extracted from the trainer's stdout by a script that's since been retired —
the CSVs are the committed record now, not regenerable from this repo alone.

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

Published checkpoints, in HF format, converted from the raw `es-at-scale`
checkpoints. `es-at-scale` saves a raw state_dict straight from vLLM's
internal Qwen2 model, which fuses attention QKV into one `qkv_proj` tensor
and MLP gate+up into one `gate_up_proj` tensor — HF's `AutoModelForCausalLM`
expects those split back into `q_proj`/`k_proj`/`v_proj` and
`gate_proj`/`up_proj` (every other parameter name matches already). The
converter that did this split was Qwen2.5-specific and isn't part of this
repo — write your own for whatever architecture you're training if you need
this step; it's a few dozen lines given the mapping above.

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

## Planned: matched-budget ES vs. RL

**Not yet run.** The single biggest weakness in this repo is that every
ES-vs-RL number compares an ES run to an already-fully-trained public
checkpoint. This section fixes what "matched" should mean before spending money
on it, because the obvious answers conflict.

### Reshape GRPO to ES's geometry, don't reshape ES

Holding the published recipes, matching on one axis un-matches the others:

| Axis to match | ES config needed | Exposures | Epochs | Generations |
|---|---|---|---|---|
| **Generations** (819,200) | 100 iters, batch 256, pop 32 | 25,600 | 3.00 | 819,200 ✅ |
| **Data** (102,400 exposures) | 400 iters, batch 256, pop 32 | 102,400 ✅ | 12.01 ✅ | 3,276,800 (4x RL) |
| **Both** | 100 iters, batch 1024, **pop 8** | 102,400 ✅ | 12.01 ✅ | 819,200 ✅ |

The third row is a trap. Exact dual parity by moving *ES* has one solution at
100 steps and it is **population 8** — a tiny sample for a direction in 7.6e9
dimensions, with binary reward over 1,024 prompts making the fitness ties that
rank shaping depends on much worse. It would test a broken ES, not ES.

**Move GRPO instead.** Retrain the RL arm at **batch 256, 32 samples per
prompt** — ES's exact geometry — and every axis matches at once, with neither
method pushed anywhere strange:

| | Steps | Prompts/step | ×per prompt | Exposures | Epochs | Generations |
|---|---|---|---|---|---|---|
| ES iter50 | 50 | 256 | 32 population | 12,800 | 1.50 | 409,600 |
| **GRPO reshaped** | 50 | 256 | 32 rollouts | 12,800 | 1.50 | 409,600 |
| ES iter100 | 100 | 256 | 32 population | 25,600 | 3.00 | 819,200 |
| **GRPO reshaped** | 100 | 256 | 32 rollouts | 25,600 | 3.00 | 819,200 |

### Does reshaping break GRPO? Two effects, opposite signs

**Group size 8 → 32 helps, and at 1.5B it helps a lot.** GRPO's advantage is
`(r_i − mean(r_group)) / std(r_group)`. Under a binary reward, a group whose
rollouts all agree has zero advantage everywhere — that prompt contributes
*nothing*, and its entire generation budget is wasted. P(degenerate) = pᴳ +
(1−p)ᴳ:

| true pass rate | G=8 | G=32 |
|---|---|---|
| 0.01 / 0.99 | 0.92 | 0.73 |
| 0.05 / 0.95 | 0.66 | 0.19 |
| 0.10 / 0.90 | 0.43 | 0.03 |
| 0.50 | 0.01 | 0.00 |

How much that matters depends on where the model sits. Using each model's
*observed* iteration-1 mean training reward and a Beta difficulty spread, the
fraction of prompts yielding usable gradient signal:

| Model | Mean reward | G=8 | G=32 | Gain |
|---|---|---|---|---|
| 1.5B | 0.014 | 0.15 – 0.25 | 0.33 – 0.42 | **1.7 – 2.2x** |
| 7B | 0.457 | 0.77 – 0.95 | 0.94 – 1.00 | 1.1 – 1.2x |

At 1.5B, where nearly every training problem is beyond the base model, G=8
wastes most of its budget on all-wrong groups and G=32 roughly doubles the
usable signal. At 7B most prompts are mid-difficulty already, so the gain is
small. Group-mean standard error also halves (0.177 → 0.088 at p=0.5).

**Prompt batch 1024 → 256 hurts.** Four times fewer distinct problems per
optimizer step. The 32 rollouts within a prompt are correlated, so effective
problem diversity tracks the *prompt* count, not the generation count — this is
the real cost, and it is why published work finds an optimum group size rather
than "bigger is better" under a fixed rollout budget.

**Net expectation: roughly a wash at 7B, plausibly net-positive at 1.5B.** Not a
drastic change in either direction. Two supporting points: 32 is unremarkable
as a group size — DeepSeekMath used G=64 at 7B and DAPO uses 512×16, so
SimpleRL-Zoo's 8 is on the *small* end and this moves toward the mainstream, not
away from it. And at equal generation count, 256×32 is *cheaper* in wall-clock
than 1024×8, because vLLM prefix-caches each shared prompt across 32 samples
instead of 8.

### The protocol

1. **Train our own GRPO arm** at batch 256 / 32 rollouts, rather than using the
   public checkpoint — the only way to control the budget. Also set
   `max_response_length=2048` to match ES and the eval, removing the token-cap
   confound in the same stroke.
2. **Run it twice: 50 and 100 steps**, giving exact parity with ES iter50 and
   ES iter100 respectively. The 100-step arm is the headline comparison.
3. **Keep the published SimpleRL-Zoo checkpoint as a separate reference row**,
   clearly labelled as fully-trained (12.01 epochs, 1024×8, 8192 tokens). Two
   RL points — one budget-matched, one fully-trained — is what separates "ES vs
   RL as methods" from "ES vs a model that saw 8x the data".
4. **Record wall-clock and total generated tokens alongside generations.**
   Generation parity is not FLOP parity: GRPO adds a backward pass ES does not
   have. Report all three and let the reader choose a denominator.
5. **Run the existing pass@k suite unchanged**, so the new arms drop straight
   into the tables already here.

**Caveat to carry:** the budget-matched arm sees 1.50 (or 3.00) epochs against
the published checkpoint's 12.01, so it will almost certainly score *below*
published SimpleRL-Zoo. That is the intended result of a matched budget, not a
failed reproduction — but it means this arm must never be described as
reproducing their numbers. Learning rate may also want revisiting at 4x the
smaller prompt batch.

### What would falsify what

- If ES-at-matched-generations still shows the 7B pass@k narrowing, the
  narrowing is a property of ES, not of its small data budget.
- If it does not, the Experiment 2 result was a budget artifact and the
  headline claim needs revising.
- If GRPO-at-2048-tokens loses its pass@1 advantage, part of the published
  ES-vs-RL gap was the token cap all along.

## Planned: full-MMLU forgetting measurement

**Not yet run.** All post-training here is math-only, so the natural question
is what it costs elsewhere. GPQA-diamond
([above](#out-of-domain-check-gpqa-diamond)) is the only out-of-domain
evidence in the repo, and it is thin for the job: 198 questions, one domain,
and a 1.5B base model sitting at chance (26.3% vs 25%), which makes its one
large result impossible to attribute.

Full MMLU is the standard instrument for this and fixes both problems.

| Parameter | Value | Why |
|---|---|---|
| Task | `mmlu` (all 57 subjects, **14,042** test questions) | 71x GPQA's question count; the CI shrinks by ~8x. lm_eval reports the aggregate, the 4 categories, and every subject from one run. |
| Shots | **5** | MMLU's reporting convention, and it lifts the base model off the chance floor that makes the GPQA number ambiguous. Shots come from the dev split and are identical across arms. |
| Scoring | log-likelihood over `["A","B","C","D"]` | Same probe class as GPQA. **Nothing is generated** — `max_tokens` and `temperature` do not apply. |
| Arms | 1.5B base/ES/RL, 7B base/ES/ES-iter100/RL | Base is the reference for forgetting; RL is the control that says whether any math post-training does this, or only ES. |
| Statistic | paired McNemar + 95% CI vs. each arm's own base | Forgetting is a within-question claim, so it needs the paired test, not a difference of two accuracies. |

Driver: not yet written — will be built alongside the GPQA redo
([Later](#later)), on the same `lm_eval`/vLLM pattern. Runs on a single GPU;
no rented box needed, unlike the pass@k suite.

**Read-out.** *Catastrophic forgetting* would be a significant drop vs. base
concentrated in the non-STEM categories (humanities, social sciences) while
math-adjacent STEM holds or rises — that is the signature of a math-only
objective overwriting unrelated capability. A uniform drop across all four
categories is degradation, not forgetting. No significant movement anywhere is
the null, and given 14,042 questions it would be a genuinely tight null rather
than the underpowered kind.

The same run also settles the GPQA question. If ES's advantage appears only
where the base is at chance, it is option-scoring calibration; if it survives
5-shot on a base scoring ~50%, it is something real. Running the sweep a second
time at 0-shot isolates that directly, on identical questions, for one extra
GPU-hour.

## Later

- **Redo the GPQA sweep, and build the MMLU one alongside it.** The
  `lm_eval`-based sweep script and its reduced CSVs were removed during a
  repo cleanup pass; rewriting it is a prerequisite for both the GPQA
  replication below and the [full-MMLU protocol](#planned-full-mmlu-forgetting-measurement)
  above.
- **Chase the 1.5B GPQA gain.** +8.6 points out-of-domain from math-only ES
  training (p = 0.008) is the most interesting unexplained result in the repo,
  and it rests on 198 questions against a base model at chance. The
  [full-MMLU protocol](#planned-full-mmlu-forgetting-measurement) below is
  designed to settle it; a second training seed is the other half of the answer.
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
- **Recompute the 7B base/ES question-level breakdowns.** The first eval box
  was torn down without keeping any per-question scores, so `results/` holds
  the aggregate pass@k curves and the base-vs-ES four-way counts but nothing
  that can be re-paired against a new arm. Raw generation trees (`base/`,
  `trained/`, `rl/`) are gitignored and local-only, so this keeps recurring
  unless a box's `results/<run_tag>/` is copied off before the box is
  released — there's currently no script that reduces them to something
  committable.
- A compute-matched RL baseline (capped at similar wall-clock/FLOP budget to
  the ES run, rather than comparing against an already fully-trained public
  checkpoint) would make the ES-vs-RL comparison fair — currently open at both
  scales.
- A fourth arm testing EGGROLL (Sarkar et al., arXiv:2511.16652 — rank-r
  LoRA-factorized ES, as opposed to `es-at-scale`'s full-rank ES).
