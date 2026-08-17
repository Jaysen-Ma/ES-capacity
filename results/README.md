# Results

The full numerical record behind the plots and claims in the
[top-level README](../README.md). Same numbers, more digits — the main README
shows the curves and the conclusions, this file is the lookup table.

Two kinds of evaluation live here and they are not comparable to each other:

- **pass@k on the math suite** — generative. Unbiased pass@k estimates (Chen et
  al., 2021) over `n_sampling` completions per question, at identical
  generation settings for every arm: `temperature=0.6`, `top_p=0.95`,
  `max_tokens=2048`, `qwen-boxed` template, `seed=1`.
- **GPQA-diamond** — out-of-domain, `lm_eval`, multiple-choice scored
  by log-likelihood over the answer options. **Nothing is generated**, so
  `max_tokens`/`temperature` do not apply and have no counterpart here.
  **Paused pending a redo** — see [GPQA-diamond](#gpqa-diamond-out-of-domain)
  below.

## Layout

One directory per training run, named `<model>-sigma<σ>-iter<N>`. Every run has
the two log-derived files; only the two headline runs have pass@k.

| Path | What |
|---|---|
| `<run>/training_curves.csv` | Per-iteration population reward (mean/std/min/max) + wall time. Present for all 7 runs. |
| `<run>/inloop_eval.csv` | The trainer's own eval-suite pass@1, tagged `baseline` or `iter<N>`. Not comparable to the pass@k numbers below — different harness and sampling settings. |
| `1.5b-sigma001-iter50/*_threeway_summary.json`, `*_threeway_passk.png` | Experiment 1 — base vs. ES vs. RL |
| `7b-sigma001-iter50/*_threeway_summary.json`, `*_threeway_passk.png` | Experiment 2 — base vs. ES vs. RL |

Experiment 1's `wandb_curves.csv` and its four `train_*.png` plots (the only
run with response-length data) are gitignored, kept local only — no script
in this repo regenerates the plots from the CSV anymore.

GPQA-diamond results (`gpqa/scores.csv`, `per_question.csv`, `mcnemar.csv`) existed here for one sweep and have been removed pending a redo — see the main README's [GPQA section](../README.md#out-of-domain-check-gpqa-diamond).

The runs, and which evaluations each one has:

| Run | pass@k suite | Checkpoint published |
|---|---|---|
| `1.5b-sigma001-iter50` | yes (3-way) | yes |
| `7b-sigma001-iter50` | yes (3-way) | yes |
| `7b-sigma001-iter100` | no | yes ([zocrate/Qwen2.5-7B-ES-math-iter100](https://huggingface.co/zocrate/Qwen2.5-7B-ES-math-iter100)) |
| `7b-sigma0025-iter50` | no | no |
| `7b-sigma0025-b64-aborted` | no | no |
| `7b-sigma005-b64-aborted` | no | no |
| `7b-sigma01-b64-aborted` | no | no |

### What is gitignored, and how to regenerate it

Raw dumps stay out of git; everything above is derived from them and committed.

| Ignored | Regenerate with | Reduced to |
|---|---|---|
| `<run>/{base,trained,rl}/` — raw per-shard generations, every completion in full (GBs) | `scripts/run_eval.sh`, once per arm (~3h35m for the 7B suite's base+trained) | `*_summary.json` + `*_passk.png` via `scripts/analyze_passk.py`, which reads these directly — nothing reduces them for git, so a torn-down box's per-question record is gone with it |
| `gpqa_results/` — the `lm_eval` tree. Its `samples_*.jsonl` embed verbatim GPQA text including presigned S3 URLs that trip GitHub secret scanning. | sweep script removed pending a redo (see main README) | n/a for now |
| `logs/` — raw trainer stdout, 7 runs, ~1.6MB | n/a (kept locally; the 7B runs used `--logging none`, so this is their only per-iteration record) | `<run>/training_curves.csv`, `<run>/inloop_eval.csv`, via a script that's since been retired — the CSVs are the committed record now |

## pass@k

### Experiment 1 — Qwen2.5-1.5B (base vs. ES vs. RL)

**AIME24** (30 questions, n_sampling=512)

| k | Base | ES | RL |
|---|---|---|---|
| 1 | 0.28% | 0.87% | 0.87% |
| 2 | 0.55% | 1.68% | 1.69% |
| 4 | 1.08% | 3.14% | 3.16% |
| 8 | 2.05% | 5.57% | 5.63% |
| 16 | 3.74% | 9.04% | 9.27% |
| 32 | 6.38% | 13.09% | 14.03% |
| 64 | 10.01% | 17.45% | 20.10% |
| 128 | 14.59% | 22.91% | 28.41% |
| 256 | 19.33% | 30.00% | 39.39% |
| 512 | 23.33% | 36.67% | 50.00% |

**MATH500** (500 questions, n_sampling=128)

| k | Base | ES | RL |
|---|---|---|---|
| 1 | 5.37% | 16.79% | 13.35% |
| 2 | 9.90% | 27.79% | 23.28% |
| 4 | 17.53% | 41.69% | 37.37% |
| 8 | 28.93% | 56.05% | 53.52% |
| 16 | 43.27% | 68.24% | 67.70% |
| 32 | 57.76% | 77.26% | 77.81% |
| 64 | 69.76% | 83.92% | 84.71% |
| 128 | 78.60% | 89.20% | 90.00% |

**Minerva Math** (272 questions, n_sampling=128)

| k | Base | ES | RL |
|---|---|---|---|
| 1 | 1.79% | 2.83% | 3.72% |
| 2 | 3.41% | 5.23% | 6.79% |
| 4 | 6.26% | 9.13% | 11.62% |
| 8 | 10.80% | 14.75% | 18.19% |
| 16 | 17.15% | 21.65% | 25.74% |
| 32 | 24.59% | 28.99% | 33.42% |
| 64 | 32.41% | 36.43% | 40.68% |
| 128 | 40.44% | 43.75% | 47.43% |

**OlympiadBench** (675 questions, n_sampling=128)

| k | Base | ES | RL |
|---|---|---|---|
| 1 | 2.10% | 6.17% | 5.41% |
| 2 | 3.96% | 10.60% | 9.67% |
| 4 | 7.18% | 16.84% | 16.04% |
| 8 | 12.21% | 24.43% | 24.12% |
| 16 | 19.10% | 32.43% | 32.64% |
| 32 | 27.24% | 40.12% | 40.60% |
| 64 | 35.86% | 47.36% | 47.84% |
| 128 | 44.74% | 54.22% | 54.22% |

### Experiment 2 — Qwen2.5-7B (base vs. ES vs. RL)

**AIME24** (30 questions, n_sampling=512)

| k | Base | ES | RL | ES − base | RL − base |
|---|---|---|---|---|---|
| 1 | 7.14% | 7.98% | 15.39% | +0.85 | +8.26 |
| 2 | 11.39% | 12.42% | 19.13% | +1.03 | +7.74 |
| 4 | 16.60% | 17.87% | 22.71% | +1.27 | +6.11 |
| 8 | 22.41% | 23.67% | 27.12% | +1.26 | +4.71 |
| 16 | 28.44% | 29.34% | 32.89% | +0.90 | +4.45 |
| 32 | 34.72% | 35.41% | 39.41% | +0.69 | +4.69 |
| 64 | 42.13% | 42.14% | 45.74% | +0.01 | +3.61 |
| 128 | 51.42% | 49.98% | 50.94% | -1.44 | -0.48 |
| 256 | 62.62% | 58.44% | 54.44% | -4.17 | -8.18 |
| 512 | 76.67% | 66.67% | 56.67% | -10.00 | -20.00 |

**MATH500** (500 questions, n_sampling=128)

| k | Base | ES | RL | ES − base | RL − base |
|---|---|---|---|---|---|
| 1 | 61.15% | 67.54% | 76.04% | +6.39 | +14.89 |
| 2 | 73.08% | 77.07% | 81.14% | +3.99 | +8.06 |
| 4 | 81.05% | 83.37% | 84.95% | +2.32 | +3.89 |
| 8 | 86.42% | 87.90% | 87.81% | +1.48 | +1.39 |
| 16 | 90.17% | 91.14% | 89.98% | +0.97 | -0.19 |
| 32 | 92.78% | 93.24% | 91.55% | +0.45 | -1.24 |
| 64 | 94.43% | 94.55% | 92.71% | +0.12 | -1.72 |
| 128 | 95.40% | 95.80% | 93.80% | +0.40 | -1.60 |

**Minerva Math** (272 questions, n_sampling=128)

| k | Base | ES | RL | ES − base | RL − base |
|---|---|---|---|---|---|
| 1 | 23.14% | 26.99% | 35.38% | +3.85 | +12.24 |
| 2 | 32.21% | 35.60% | 40.69% | +3.39 | +8.48 |
| 4 | 40.94% | 42.90% | 45.27% | +1.97 | +4.33 |
| 8 | 48.24% | 48.77% | 49.44% | +0.53 | +1.20 |
| 16 | 54.10% | 53.78% | 53.06% | -0.32 | -1.04 |
| 32 | 58.81% | 58.26% | 56.10% | -0.55 | -2.71 |
| 64 | 62.57% | 62.29% | 58.92% | -0.29 | -3.65 |
| 128 | 65.81% | 65.81% | 62.13% | +0.00 | -3.68 |

**OlympiadBench** (675 questions, n_sampling=128)

| k | Base | ES | RL | ES − base | RL − base |
|---|---|---|---|---|---|
| 1 | 27.95% | 32.39% | 38.59% | +4.44 | +10.64 |
| 2 | 37.10% | 41.05% | 45.13% | +3.95 | +8.04 |
| 4 | 45.12% | 48.50% | 51.24% | +3.38 | +6.11 |
| 8 | 52.25% | 55.12% | 56.82% | +2.88 | +4.57 |
| 16 | 58.89% | 61.21% | 61.86% | +2.32 | +2.98 |
| 32 | 64.91% | 66.92% | 66.48% | +2.01 | +1.56 |
| 64 | 70.09% | 72.00% | 70.58% | +1.91 | +0.48 |
| 128 | 74.67% | 76.30% | 73.78% | +1.63 | -0.89 |

## GPQA-diamond (out-of-domain)

Paused pending a redo — the sweep script and reduced CSVs (`gpqa/*.csv`) have
been removed from this repo. See the main README's
[GPQA section](../README.md#out-of-domain-check-gpqa-diamond) for a summary
of what an earlier sweep found and why it's being re-run rather than kept as
the record.

## Question-coverage breakdown

"Solvable" = at least 1 of `n_sampling` completions correct. *narrow* = solvable
by base but not by the trained model; *gain* = the reverse.

Experiment 1 (1.5B):

| Benchmark | ES: narrow / gain / **net** | RL: narrow / gain / **net** |
|---|---|---|
| AIME24 | 0.0% / 13.3% / **+13.3** | 3.3% / 30.0% / **+26.7** |
| MATH500 | 1.4% / 12.0% / **+10.6** | 1.2% / 12.6% / **+11.4** |
| Minerva | 7.0% / 10.3% / **+3.3** | 5.5% / 12.5% / **+7.0** |
| OlympiadBench | 3.9% / 13.3% / **+9.4** | 3.4% / 12.9% / **+9.5** |

Experiment 2 (7B):

| Benchmark | pass@1 base→ES→RL | pass@max base→ES→RL | ES: narrow / gain / **net** | RL: narrow / gain / **net** |
|---|---|---|---|---|
| AIME24 (k≤512) | 7.14%→7.98%→15.39% | 76.67%→66.67%→56.67% | 13.3% / 3.3% / **−10.0** | 23.3% / 3.3% / **−20.0** |
| MATH500 (k≤128) | 61.15%→67.54%→76.04% | 95.40%→95.80%→93.80% | 1.0% / 1.4% / **+0.4** | 3.0% / 1.4% / **−1.6** |
| Minerva (k≤128) | 23.14%→26.99%→35.38% | 65.81%→65.81%→62.13% | 2.9% / 2.9% / **0.0** | 5.1% / 1.5% / **−3.6** |
| OlympiadBench (k≤128) | 27.95%→32.39%→38.59% | 74.67%→76.30%→73.78% | 3.4% / 5.0% / **+1.6** | 5.5% / 4.6% / **−0.9** |

## Training runs

All 7 runs, from `<run>/training_curves.csv`. Population 32,
`max_tokens=2048`, seed 42 throughout; the 1.5B run is Qwen2.5-1.5B and every
other run is Qwen2.5-7B. Reward is the population mean of the binary math
reward. `Σ iter wall` sums per-iteration times and so excludes startup and
model loading, which is why it runs slightly under the wall-clock figures in
the main README.

| Run | σ | α | Batch | Iterations | Reward start | Reward end | Reward peak | Σ iter wall |
|---|---|---|---|---|---|---|---|---|
| `1.5b-sigma001-iter50` | 0.001 | auto (σ/2) | 256 | 1–51 | 0.014 | 0.428 | 0.479 @ i45 | 3.19h |
| `7b-sigma001-iter50` | 0.001 | auto (σ/2) | 256 | 1–51 | 0.457 | 0.654 | 0.702 @ i25 | 4.34h |
| `7b-sigma001-iter100` | 0.001 | auto (σ/2) | 256 | 52–101 | 0.664 | 0.645 | 0.737 @ i75 | 4.20h |
| `7b-sigma0025-iter50` | 0.0025 | σ/4 | 256 | 1–51 | 0.355 | 0.533 | 0.649 @ i34 | 4.79h |
| `7b-sigma0025-b64-aborted` | 0.0025 | σ/4 | 64 | 1–24 | 0.423 | 0.383 | 0.511 @ i22 | 1.31h |
| `7b-sigma005-b64-aborted` | 0.005 | σ/4 | 64 | 1–2 | 0.001 | 0.000 | 0.001 @ i1 | 0.12h |
| `7b-sigma01-b64-aborted` | 0.01 | σ/4 | 64 | 1–3 | 0.000 | 0.000 | 0.000 @ i1 | 0.18h |

## In-loop eval suite

The trainer's own eval (single-sample pass@1, its own prompt/sampling
settings) — **not** comparable to the pass@k tables above. From
`<run>/inloop_eval.csv`. `baseline` is the pass over the starting weights,
which for `7b-sigma001-iter100` means the iter-50 checkpoint it resumed from.

| Run | Eval point | AIME | AMC | MATH500 | MINERVA | OLYMPIAD_BENCH |
|---|---|---|---|---|---|---|
| `1.5b-sigma001-iter50` | baseline | 0.00% | 0.00% | 1.40% | 7.35% | 0.59% |
| `1.5b-sigma001-iter50` | iter50 | 0.00% | 21.69% | 53.00% | 16.54% | 17.78% |
| `7b-sigma001-iter50` | baseline | 16.67% | 34.94% | 46.80% | 20.96% | 28.89% |
| `7b-sigma001-iter50` | iter50 | 6.67% | 40.96% | 71.40% | 34.56% | 37.19% |
| `7b-sigma001-iter100` | baseline | 6.67% | 34.94% | 73.00% | 37.87% | 36.15% |
| `7b-sigma001-iter100` | iter100 | 6.67% | 37.35% | 73.40% | 37.87% | 36.00% |
| `7b-sigma0025-iter50` | iter50 | 6.67% | 34.94% | 72.00% | 33.82% | 36.44% |
| `7b-sigma01-b64-aborted` | baseline | 16.67% | 34.94% | 46.60% | 19.85% | 30.37% |

Two pairs of rows above score **identical weights** on separate occasions, and
so measure this suite's run-to-run reproducibility directly:

- pristine Qwen2.5-7B base — `7b-sigma001-iter50` @ `baseline` vs.
  `7b-sigma01-b64-aborted` @ `baseline`
- the Experiment 2 checkpoint — `7b-sigma001-iter50` @ `iter50` vs.
  `7b-sigma001-iter100` @ `baseline`

Between 1 and 10 questions flip per benchmark per pair; AIME is identical in
both. Full breakdown and what it does and doesn't license: see the main README.

