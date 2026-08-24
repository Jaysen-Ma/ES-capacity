# GPQA-diamond out-of-domain check

Supporting detail for the [GPQA section of the README](../README.md#out-of-domain-check-gpqa-diamond).
The headline is there: both base models sit at four-choice chance, so this
benchmark has no headroom in which ES or RL training could be shown to have
degraded anything. This page is the method and the full numbers behind that.

## What was run

GPQA-diamond zero-shot, 198 graduate-level science questions, four choices
each, through `lm_eval` 0.4.12 with a vLLM backend. Scoring is log-likelihood
over the four options — the model never generates an answer, it just ranks the
four strings — so this is a different measurement from the pass@k work in the
README, which samples completions and grades them.

Six arms: `Qwen2.5-1.5B` and `Qwen2.5-7B`, each as base, ES-trained and the
SimpleRL-Zoo GRPO checkpoint. Every arm was scored under 10 answer-order
permutations (seeds 0-9). Hardware was a single GB10 (sm_121 via PTX
forward-compatibility), which is why this probe could run locally while the
pass@k suite could not.

## Why the answer order is swept

Log-likelihood scoring is deterministic. Running the same model on the same
questions twice gives a byte-identical number, so repetition measures nothing
and there is no error bar to be had from re-running.

What does vary is the order of the four answer choices. `lm_eval`'s GPQA task
shuffles them with a plain `random.shuffle` on the global RNG, so the seed
decides which slot the correct answer lands in. For a model near chance, that
is not a nuisance parameter — it is the dominant one, because a model with any
preference among the four slots will score differently depending on where the
correct answers happen to sit. Sweeping the seed turns one draw into a
distribution.

Two things have to hold for the sweep to mean anything, and
`scripts/analyze_gpqa.py --check` refuses to report unless both do:

1. **Within a seed, all six arms must see the identical shuffle**, checked by
   comparing the per-question `doc_hash` across arms. Otherwise pairing them is
   meaningless.
2. **Across seeds, the shuffles must actually differ.** The Hugging Face
   `datasets` cache is keyed on a fingerprint of the map function and the
   dataset, and never on the RNG state, so with caching left on every seed after
   the first silently reuses the first permutation. That failure produces ten
   identical results and no error at all, which is why it is checked explicitly.
   `scripts/run_gpqa_sweep.py` calls `datasets.disable_caching()` before any task
   is built.

## Every arm sits at chance

| Arm | mean acc over 10 permutations | sd | min | max |
|---|---|---|---|---|
| 1.5B-base | 23.79% | 2.23 | 20.71% | 28.79% |
| 1.5B-ES | 25.05% | 3.12 | 20.20% | 29.29% |
| 1.5B-RL | 23.69% | 2.81 | 18.18% | 28.28% |
| 7B-base | 25.45% | 2.53 | 21.72% | 27.78% |
| 7B-ES | 25.00% | 1.91 | 22.22% | 28.28% |
| 7B-RL | 26.01% | 2.61 | 21.72% | 28.79% |

Chance is 25%. No arm is more than 1.3 points from it, and every arm's
permutation spread (min to max) is 6 to 9 points wide. Reshuffling the answers
moves one model further than any two models are apart.

## Every comparison is null

Paired within a seed, so both arms answered the identically-shuffled questions.
The p-value is a Wilcoxon signed-rank test over per-question outcomes pooled
across the ten permutations.

| Comparison | mean delta (pts) | range | arm ahead on | Wilcoxon p |
|---|---|---|---|---|
| 1.5B-ES vs 1.5B-base | **+1.26 ± 3.20** | -3.54 … +7.07 | 7/10 | 0.2479 |
| 1.5B-RL vs 1.5B-base | **-0.10 ± 1.86** | -2.53 … +2.53 | 4/10 | 0.7892 |
| 1.5B-ES vs 1.5B-RL | **+1.36 ± 2.01** | -1.01 … +6.06 | 7/10 | 0.2346 |
| 7B-ES vs 7B-base | **-0.45 ± 2.34** | -4.04 … +3.03 | 5/10 | 0.7338 |
| 7B-RL vs 7B-base | **+0.56 ± 0.73** | +0.00 … +2.02 | 5/10 | 0.147 |
| 7B-ES vs 7B-RL | **-1.01 ± 2.56** | -5.56 … +3.03 | 4/10 | 0.2801 |

Nothing here clears significance, and the mean deltas are all small next to
their own spread across permutations.

## What a single permutation would have told you instead

Take the largest of those comparisons, 1.5B ES against 1.5B base, and read it
one permutation at a time. The gained/lost columns are questions the arm gets
right that the base gets wrong, and the reverse; the p-value is an exact
McNemar test on that pair.

| seed | delta | gained / lost | McNemar p |
|---|---|---|---|
| 0 | +0.51 | 17 / 16 | 1.00 |
| **1** | **+7.07** | **20 / 6** | **0.0094** |
| 2 | −0.51 | 13 / 14 | 1.00 |
| 3 | +1.52 | 21 / 18 | 0.75 |
| 4 | +5.05 | 18 / 8 | 0.076 |
| 5 | +2.53 | 19 / 14 | 0.49 |
| 6 | −2.53 | 10 / 15 | 0.42 |
| 7 | −3.54 | 15 / 22 | 0.32 |
| 8 | +2.02 | 12 / 8 | 0.50 |
| 9 | +0.51 | 13 / 12 | 1.00 |

One permutation in ten clears p<0.05, at +7.07 points with 20 questions gained
against 6 lost. That looks like a solid out-of-domain gain, and it is the same
198 questions and the same two sets of weights as seed 7, which comes out at
−3.54. Reporting a single GPQA number for a near-chance model is reporting the
shuffle, not the model. This is why the sweep exists.

## Where the arms do differ: answer-position preference

GPQA places the correct answer uniformly across the four slots, so a model with
no positional preference picks each slot about 25% of the time. Pooled over all
ten permutations:

| Arm | picks (A) | (B) | (C) | (D) | spread |
|---|---|---|---|---|---|
| 1.5B-base | 35.8% | 18.9% | 16.4% | 28.9% | 19.3 pts |
| 1.5B-ES | 20.1% | 24.9% | 35.7% | 19.4% | 16.3 pts |
| 1.5B-RL | 35.4% | 22.2% | 16.1% | 26.4% | 19.2 pts |
| 7B-base | 27.8% | 23.4% | 22.7% | 26.2% | 5.1 pts |
| 7B-ES | 36.2% | 22.6% | 9.7% | 31.5% | 26.6 pts |
| 7B-RL | 26.8% | 22.8% | 23.3% | 27.1% | 4.3 pts |

None of these arms is answering from science knowledge, so what the scoring
picks up is which answer letter each one puts probability mass on. The
preferences are strong and they differ between arms: at 1.5B, base and RL both
favour A while ES favours C; at 7B the base is nearly flat at a 5-point spread
while ES is at 26.6.

That is the mechanism behind the permutation sensitivity above. A shuffle that
seats more correct answers in C flatters 1.5B ES; one that seats them in A
flatters 1.5B base. It is a property of how these models rank four candidate
strings, not a capability, and it is a side observation rather than a result
this project set out to measure.

## Limits

**Absolute levels are harness- and hardware-specific.** These 198 questions are
often decided by small log-probability margins between the four options, so
kernel-level numeric differences flip many items. The same `Qwen2.5-7B` weights
from the same Hub snapshot score 32.8% under an earlier `lm_eval`/vLLM on 8x
RTX 4090 and 25.5 ± 2.5 here on the GB10 — a gap the permutation spread does not
explain. Within this sweep every arm shares one harness, one GPU and the same
ten permutations, so the deltas above are sound. Comparing these absolute
numbers against a GPQA number from anywhere else is not.

**This is one 198-question benchmark scored one way.** It says these six models
are indistinguishable from each other and from guessing on graduate science
multiple-choice. It does not measure forgetting broadly, and with both bases at
chance it could not have, whatever the training did. That measurement needs a
benchmark the bases score above chance on, which is the still-open MMLU
protocol in the README.

## Reproduce

    python scripts/run_gpqa_sweep.py                 # 6 arms x 10 permutations
    python scripts/analyze_gpqa.py --sweep-root results/gpqa_results \
        --out results/gpqa \
        --pair 1.5B-ES:1.5B-base --pair 1.5B-RL:1.5B-base \
        --pair 7B-ES:7B-base   --pair 7B-RL:7B-base

The reduced CSVs are committed in `results/gpqa/`:

| File | Contents |
|---|---|
| `scores.csv` | accuracy per (seed, arm) |
| `scores_summary.csv` | per-arm mean/sd/min/max across the 10 permutations |
| `per_question.csv` | doc_id x arm, correct on how many seeds |
| `mcnemar.csv` | exact paired test per (seed, comparison) |
| `pairs_summary.csv` | delta distribution + Wilcoxon per comparison |
| `position_bias.csv` | slot A/B/C/D pick rates, pooled over seeds |

Every table on this page and in the README's GPQA section is generated from
those CSVs by `python scripts/gpqa_tables.py`, so they can be diffed against
the artifacts rather than trusted.

The raw `lm_eval` tree under `results/gpqa_results/` is gitignored: its
`samples_*.jsonl` are ~2.6 MB per arm per seed and embed the verbatim benchmark
questions, including presigned S3 URLs that trip GitHub secret scanning.
