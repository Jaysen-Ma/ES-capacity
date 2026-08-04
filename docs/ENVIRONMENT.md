# Environments for ES-capacity

Torch, transformers, vLLM, Ray, and Triton are **machine-provided** — not pinned in `pyproject.toml`. No single lockfile spans aarch64/CUDA 13 and generic x86 CUDA.

## This box: GB10 (aarch64, CUDA 13)

Verified working stack (shared jupyterlab venv as reference; prefer a dedicated `.venv`):

| Package | Version noted |
|---------|----------------|
| Python | 3.12.3 |
| torch | 2.11.0+cu130 |
| transformers | 5.x |
| vllm | 0.26.0 |
| triton | 3.6.0 |
| peft | 0.19.x |

```bash
cd /path/to/ES-capacity
python3 -m venv .venv
source .venv/bin/activate
# Install torch/vllm the same way you provisioned your GB10 wheels
# (GB10 wheels — do not pip install generic cu12 wheels)
pip install -e .
# Grading pins from pyproject: sympy==1.12, antlr4-python3-runtime==4.11.1
cd third_party/yue_math/latex2sympy && pip install -e . --use-pep517 && cd -
```

If you keep a venv outside the repo (e.g. a shared one), point `paths.venv` in `config.local.toml` at it, or export `ES_CAPACITY_VENV=/path/to/venv`; `scripts/*.sh` pick it up automatically via `scripts/_env.sh` without any script edits.

**Quirks**

- Compute capability **12.1**; torch arch list has sm_80/90/100/110/120 but **not sm_121** → forward-compat/PTX.
- Multi-LoRA for EGGROLL depends on Triton JIT (present). Probe before training.
- `/dev/shm` ≈ 61 GB — enough for large LoRA populations.
- Docker fallback: `vllm/vllm-openai:gemma4-cu130`.

## Generic x86 CUDA (reproducibility for others)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cu128
pip install vllm ray
pip install -e .
```

Use `configs/machine/example.toml` as a starting machine profile — copy it to `configs/machine/<name>.toml` and adjust hardware fields.

## Config

```bash
cp config.local.example.toml config.local.toml
# edit models_dir, model paths (and venv, optional)
```

Merge order: `machine` ← `profile` ← `experiment` ← `config.local.toml`.

No CLI or script hardcodes a machine name. Resolution order for `--machine`:

1. Explicit `--machine <name>` flag.
2. `$ES_CAPACITY_MACHINE` env var — set this in your own shell rc (not in the repo) so your machine profile is the default everywhere.
3. `example` (generic, hardware-agnostic fallback), so a fresh clone works without any setup beyond `config.local.toml`.

`scripts/run_v1_*.sh` and `scripts/train_eggroll_7b.sh` are this project's own repro scripts for the `gb10` box; they default to `${ES_CAPACITY_MACHINE:-gb10}` rather than the generic fallback, but still respect the env var override.
