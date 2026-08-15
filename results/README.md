# Results

The full numerical record behind the plots and claims in the
[top-level README](../README.md). Same numbers, more digits — the main README
shows the curves and the conclusions, this file is the lookup table.

All figures are unbiased pass@k estimates (Chen et al., 2021) computed over
`n_sampling` completions per question, at identical generation settings for
every arm: `temperature=0.6`, `top_p=0.95`, `max_tokens=2048`, `qwen-boxed`
template, `seed=1`.

## Layout

| Path | What |
|---|---|
| `iter50-1.5b/*_threeway_summary.json`, `*_threeway_passk.png` | Experiment 1 — base vs. ES vs. RL |
| `iter50-1.5b/*_summary.json`, `*_passk.png` | Experiment 1 — base vs. ES only (superseded by the three-way files) |
| `iter50-1.5b/train_*.png`, `training_curves.csv` | Experiment 1 training dynamics |
| `iter50-7b/*_summary.json`, `*_passk.png` | Experiment 2 — base vs. ES |

Raw per-shard generations (`iter50-1.5b/{base,trained,rl}/`,
`iter50-7b/{base,trained}/`) and the GPQA `lm_eval` tree are gitignored —
regenerate them with the scripts documented in the main README.

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

### Experiment 2 — Qwen2.5-7B (base vs. ES)

**AIME24** (30 questions, n_sampling=512)

| k | Base | ES | ES − base |
|---|---|---|---|
| 1 | 7.14% | 7.98% | +0.85 |
| 2 | 11.39% | 12.42% | +1.03 |
| 4 | 16.60% | 17.87% | +1.27 |
| 8 | 22.41% | 23.67% | +1.26 |
| 16 | 28.44% | 29.34% | +0.90 |
| 32 | 34.72% | 35.41% | +0.69 |
| 64 | 42.13% | 42.14% | +0.01 |
| 128 | 51.42% | 49.98% | -1.44 |
| 256 | 62.62% | 58.44% | -4.17 |
| 512 | 76.67% | 66.67% | -10.00 |

**MATH500** (500 questions, n_sampling=128)

| k | Base | ES | ES − base |
|---|---|---|---|
| 1 | 61.15% | 67.54% | +6.39 |
| 2 | 73.08% | 77.07% | +3.99 |
| 4 | 81.05% | 83.37% | +2.32 |
| 8 | 86.42% | 87.90% | +1.48 |
| 16 | 90.17% | 91.14% | +0.97 |
| 32 | 92.78% | 93.24% | +0.45 |
| 64 | 94.43% | 94.55% | +0.12 |
| 128 | 95.40% | 95.80% | +0.40 |

**Minerva Math** (272 questions, n_sampling=128)

| k | Base | ES | ES − base |
|---|---|---|---|
| 1 | 23.14% | 26.99% | +3.85 |
| 2 | 32.21% | 35.60% | +3.39 |
| 4 | 40.94% | 42.90% | +1.97 |
| 8 | 48.24% | 48.77% | +0.53 |
| 16 | 54.10% | 53.78% | -0.32 |
| 32 | 58.81% | 58.26% | -0.55 |
| 64 | 62.57% | 62.29% | -0.29 |
| 128 | 65.81% | 65.81% | +0.00 |

**OlympiadBench** (675 questions, n_sampling=128)

| k | Base | ES | ES − base |
|---|---|---|---|
| 1 | 27.95% | 32.39% | +4.44 |
| 2 | 37.10% | 41.05% | +3.95 |
| 4 | 45.12% | 48.50% | +3.38 |
| 8 | 52.25% | 55.12% | +2.88 |
| 16 | 58.89% | 61.21% | +2.32 |
| 32 | 64.91% | 66.92% | +2.01 |
| 64 | 70.09% | 72.00% | +1.91 |
| 128 | 74.67% | 76.30% | +1.63 |

## GPQA-diamond (out-of-domain)

Zero-shot, 198 graduate-level science questions, 4-choice, scored by
log-likelihood — **not** pass@k. `lm_eval`, vLLM backend. Chance is 25%.

| Arm | acc | vs. its base | McNemar (paired, exact) |
|---|---|---|---|
| 1.5B base | 26.3% (52/198) | — | — |
| 1.5B ES | **34.8% (69/198)** | **+8.6** | 27 gained / 10 lost, **p = 0.008** |
| 1.5B RL (SimpleRL-Zoo) | 27.3% (54/198) | +1.0 | 5 gained / 3 lost, p = 0.73 |
| 7B base | 32.8% (65/198) | — | — |
| 7B ES | 31.8% (63/198) | −1.0 | 9 gained / 11 lost, p = 0.82 |
| 7B RL (SimpleRL-Zoo) | 32.8% (65/198) | ±0.0 | 2 gained / 2 lost, p = 1.00 |

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

| Benchmark | pass@1 base → ES | pass@max base → ES | narrow / gain / **net** |
|---|---|---|---|
| AIME24 (k≤512) | 7.14% → 7.98% | 76.67% → 66.67% | 13.3% / 3.3% / **−10.0** |
| MATH500 (k≤128) | 61.15% → 67.54% | 95.40% → 95.80% | 1.0% / 1.4% / **+0.4** |
| Minerva (k≤128) | 23.14% → 26.99% | 65.81% → 65.81% | 2.9% / 2.9% / **0.0** |
| OlympiadBench (k≤128) | 27.95% → 32.39% | 74.67% → 76.30% | 3.4% / 5.0% / **+1.6** |
