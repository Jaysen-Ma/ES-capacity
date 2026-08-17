# RL-arm training image — build & Vast template setup

One image, three venvs: **ES training**, **verl GRPO training**, and
**pass@k analysis**, for the ES-capacity project. See `Dockerfile` for what's
in each venv and why they can't share one environment, and `on_start.sh` for
what runs at every instance boot.

## 1. First-time setup (build + publish)

This can't be built on a Vast *instance* itself — Vast containers can't run
Docker-in-Docker. It builds in GitHub Actions instead.

```bash
git add docker/ .github/workflows/build-image.yml
git commit -m "Add RL-arm training Docker image (ES + verl + pass@k eval)"
git push
```

That triggers `.github/workflows/build-image.yml`, which builds
`docker/Dockerfile` and pushes to `ghcr.io/Jaysen-Ma/es-capacity-rl-arm:latest`.
**Expect ~1-2.5h on the first run** (the verl venv compiles flash-attn from
source); later builds reuse GitHub Actions' layer cache and are much faster
unless an earlier `Dockerfile` layer changed. Watch progress under the repo's
**Actions** tab.

GHCR packages default to **private**. Either:
- make it public: repo → **Packages** → `es-capacity-rl-arm` → *Package settings*
  → change visibility to Public (simplest — the Vast template then needs no
  registry credentials), or
- keep it private and fill in **Docker Repository Authentication** in the Vast
  template (Server `ghcr.io`, your GitHub username, and a PAT with
  `read:packages` scope).

To rebuild after editing the `Dockerfile`, just push again, or trigger
manually from the Actions tab (`workflow_dispatch`).

## 2. Vast template field values

| Field | Value |
|---|---|
| **Image Path:Tag** | `ghcr.io/Jaysen-Ma/es-capacity-rl-arm:latest` |
| **Docker Options** | `--shm-size=32g` — Ray (es-at-scale, verl) and vLLM multi-process workers need far more `/dev/shm` than Docker's 64MB default; too little causes silent worker crashes/hangs under load. |
| **Ports** | None required beyond what the base image already opens (SSH, Jupyter, Instance Portal) — training/eval run interactively, not as a served API. Optionally add `8265` if you want the Ray dashboard reachable (see base.md §7 for wiring a port through the Caddy auth edge). |
| **Environment Variables** | `NCCL_P2P_DISABLE=1` (redundant with `on_start.sh`, but set it here too so it's live before the script runs — 8x 3090/4090 boxes report `CNS` peer-to-peer topology; NCCL hangs at init without this). Optionally `HF_TOKEN` if you want checkpoints auto-pushed to HF Hub. |
| **Select Launch Mode** | Jupyter-python notebook + SSH (as already selected) — gives both a shell for an agent and a notebook if wanted. |
| **On-start Script** | Paste the full contents of `on_start.sh`. Runs on every boot; clones/pulls ES-capacity, es-at-scale, and limit-of-RLVR onto `$WORKSPACE` (idempotent — safe on stop/start too). |
| **Container disk size** | **≥80GB.** The three venvs plus the CUDA-devel base run ~40-55GB; leave headroom for pip caches, HF downloads that land outside the volume, etc. |
| **Add recommended volume settings** | Yes — **256GB** `$WORKSPACE` volume (per `RL_ARM_HANDOFF_PROMPT.md`'s storage budget: verl/vllm checkpoints, base models, ES checkpoints, eval outputs). Confirm after boot with `vast-capabilities \| jq '.instance.workspace_is_volume'` — `on_start.sh` also warns if it's `false`. |

## 3. Using the three venvs

No `source /venv/main/bin/activate` here — that env has `transformers==5.15`
/ `vllm==0.27`, incompatible with all three of these stacks.

```bash
# ES training (Jaysen-Ma/es-at-scale @ fix/multi-engine-colocation)
source /venv/es/bin/activate
cd $WORKSPACE/es-at-scale
python -m es_at_scale.train --task math --model-name Qwen/Qwen2.5-1.5B ...

# verl GRPO training (vendored at /opt/verl, pinned v0.8.0, editable-installed
# into this venv — its example configs/scripts live there too)
source /venv/verl/bin/activate
cd /opt/verl
python -m verl.trainer.main_ppo ...   # or examples/grpo_trainer/*.sh as a template

# pass@k generation + grading (Jaysen-Ma/limit-of-RLVR @ fix/math-equal-timeout-bypass)
source /venv/eval/bin/activate
cd $WORKSPACE/limit-of-RLVR/math/examples/math_eval
./run_sharded_eval.sh <model_dir> <output_dir> <benchmark> <n_sampling> ...

# ES-capacity's own wrapper scripts (eval + analysis — plain deps, /venv/main is fine)
cd $WORKSPACE/ES-capacity
scripts/run_eval.sh <model_dir> <label> <run_tag>
scripts/analyze_passk.py --run-tag <run_tag> --label ... --baseline ... --plot
```
`run_eval.sh` needs `MATH_EVAL_DIR`; since `on_start.sh` already clones
`limit-of-RLVR` onto `$WORKSPACE`, it's found automatically with no
`config.sh` needed here — that fallback only kicks in when `$WORKSPACE` is
set, which it always is on this image.

If you need to patch verl itself (e.g. to compare its math reward against
es-at-scale's, per the handoff doc's step), don't edit `/opt/verl` in place —
it's vendored in the image and not on the persistent volume, so edits vanish
on recycle. Clone your own fork onto `$WORKSPACE` and reinstall over the same
venv: `/venv/verl/bin/pip install -e . --no-deps` (deps are already satisfied
from the image build).

## 4. Updating the image later

Dependency pins (`Dockerfile`) change rarely — rebuild only when you
deliberately bump a version. Project code (ES-capacity, es-at-scale,
limit-of-RLVR) changes far more often and needs **no rebuild** — `on_start.sh`
re-syncs it on every boot.
