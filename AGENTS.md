# Agent notes for ES-capacity

## Read first

1. If `.notes/STATE.md` exists locally, **read it before doing anything**. It is the session handoff.
2. This file is committed; `.notes/` is gitignored personal state. Do not push `.notes/` or `config.local.toml`.

## Project goal (v1)

Compare three arms on Minerva Math (272 problems) via pass@k, accuracy histogram, and solvable-set coverage:

| Arm | Source |
|-----|--------|
| Base | local `Qwen2.5-7B` |
| GRPO | `hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo` (download, no training) |
| ES | EGGROLL trained locally (no public checkpoint) |

Qiu full-rank ES is a v2 drop-in behind the same `Perturber` interface.

## Layout

```
configs/machine/     committed machine profiles (gb10.toml)
configs/profiles/    smoke / pilot / v1 / scale
configs/experiments/ experiment recipes
config.local.toml    GITIGNORED absolute paths + tokens
third_party/         vendored author code (verbatim + VENDOR.md + patches/)
es_capacity/         our thin orchestration layer
data/                committed datasets + SHA256
runs/                GITIGNORED sharded eval/train outputs
.notes/              GITIGNORED session continuity
```

## Vendoring rules

- Author code under `third_party/<name>/` stays **byte-identical** except numbered `patches/*.patch` recorded in `VENDOR.md`.
- Do not refactor vendored files in place. Divergence must be a visible patch.
- Eval and train rewards must both call the same Yue grader wrapper (`es_capacity.grade`).

## Sampling artifacts

Every eval run is a sequence of fixed-size **shards**:

```
runs/<run_id>/shards/shard_NNNN/{manifest.json,completions.jsonl,records.jsonl}
runs/<run_id>/aggregate/{correct_counts.json,metrics.json}
```

- Extending `n` adds shards only; never re-generate completed shards.
- Aggregator refuses mismatched sampling params across shards.
- Comparisons truncate every arm to the minimum common `n`.

## Scale ladder

| Profile | Model | Problems | Shards |
|---------|-------|----------|--------|
| smoke | 1.5B | 8 | 1×4 |
| pilot | 7B | 32 | 1×16 |
| v1 | 7B | 272 | 4×16 (n=64) |
| scale | 7B | 272 | add shards → n=256 |

Every code change must pass `smoke` before larger runs.

## Environment

- This box: NVIDIA GB10, aarch64, CUDA 13, compute capability 12.1 (forward-compat; torch arch list omits sm_121).
- Torch / vLLM / Ray are **machine-provided** — see `docs/ENVIRONMENT.md`.
- Dedicated in-repo `.venv` preferred over shared jupyterlab venv (grader pins sympy 1.12).

## Config merge order

`configs/machine/<host>.toml` ← `configs/profiles/<profile>.toml` ← `configs/experiments/<exp>.toml` ← `config.local.toml`

No code or committed script hardcodes an absolute path or a machine name. Absolute paths (models dir, venv) live only in the gitignored `config.local.toml`. The `--machine` default resolves to `$ES_CAPACITY_MACHINE` (set in your own shell, not the repo) or else the generic `example` profile — see `docs/ENVIRONMENT.md`. `scripts/_env.sh` resolves the venv the same way (`$ES_CAPACITY_VENV` → `paths.venv` in `config.local.toml` → repo-local `.venv`) instead of any script sourcing a fixed venv path.

## Do not

- Push `.notes/`, `config.local.toml`, `runs/`, model weights.
- Edit the plan file under `~/.cursor/plans/`.
- Train GRPO in v1 (use SimpleRL-Zoo).
- Spend 7B EGGROLL compute without a rising 1.5B reward curve first.
