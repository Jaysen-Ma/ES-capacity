# ES-capacity

**Does evolution-strategy post-training expand LLM reasoning capacity beyond the base model?**

This project generalises the pass@k capacity analysis of [Yue et al.](https://arxiv.org/abs/2504.13837) from RLVR to evolution-strategy post-training, starting with [EGGROLL](https://arxiv.org/abs/2511.16652) (Sarkar et al.) and keeping [Qiu et al. full-rank ES](https://arxiv.org/abs/2509.24372) as a drop-in.

## v1 arms (Minerva Math)

| Arm | Checkpoint |
|-----|------------|
| Base | Qwen2.5-7B |
| GRPO | [hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo) |
| ES | EGGROLL trained locally (no public fine-tuned weights) |

Figures: pass@k curves, accuracy histogram, solvable-set coverage.

## Quick start

No hardcoded paths: everything machine- or user-specific lives in a config layer, not in code.

```bash
cp config.local.example.toml config.local.toml   # fill in models_dir (and venv, optional)
python -m venv .venv && source .venv/bin/activate
pip install -e .
# torch / vllm / transformers: install per docs/ENVIRONMENT.md

# Smoke (minutes) — uses configs/machine/example.toml by default:
python -m es_capacity.cli.eval --profile smoke --arm base

# Aggregate + figures:
python -m es_capacity.cli.aggregate --run-id <run_id>
python -m es_capacity.cli.figures --runs <run_a>,<run_b>
```

Everything reads through `configs/machine/<name>.toml ← configs/profiles/<profile>.toml ← configs/experiments/<exp>.toml ← config.local.toml` (see `AGENTS.md`). To adapt this to your own hardware: copy `configs/machine/example.toml` to `configs/machine/<yours>.toml`, then either pass `--machine <yours>` on every CLI call or `export ES_CAPACITY_MACHINE=<yours>` in your own shell so it becomes the default everywhere without editing any committed file.

## Layout

See [AGENTS.md](AGENTS.md) for conventions. Vendored author code lives under `third_party/` with a `VENDOR.md` and numbered patches.

## Papers

| Role | Paper | Link |
|------|-------|------|
| Capacity analysis | Yue et al. | arXiv:2504.13837 |
| Low-rank ES (EGGROLL) | Sarkar et al. | arXiv:2511.16652 |
| Full-parameter ES | Qiu et al. | arXiv:2509.24372 |
| GRPO baseline | Zeng et al. (SimpleRL-Zoo) | arXiv:2503.18892 |

## Licence

Our code is MIT (see [LICENSE](LICENSE)).

Vendored third-party code under `third_party/` keeps its **own** upstream licence, which is
not MIT and in two cases is more restrictive. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the per-tree terms before reusing anything from there — in particular, `third_party/eggroll/`
is GPL-3.0 and `third_party/qiu_es/` is restricted to non-commercial use.
