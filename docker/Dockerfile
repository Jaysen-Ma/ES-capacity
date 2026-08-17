# ES-capacity training image: ES training (es-at-scale) + verl GRPO training +
# pass@k analysis (limit-of-RLVR/math_eval), for the ES-capacity project.
#
# Three dependency stacks in this project are mutually incompatible in one
# Python env (verified against each repo's own pins as of 2026-08):
#   es-at-scale     -> transformers==4.57.6, vllm==0.11.0
#   verl (GRPO)     -> vllm>=0.8.5,<=0.12.0 (v0.8.0 tag's tested range)
#   math_eval       -> transformers<4.48,   vllm<=0.6.3
# so each gets its own venv: /venv/es, /venv/verl, /venv/eval.
#
# Repos are NOT baked in here (except a vendored verl copy — see below).
# ES-capacity, es-at-scale, and limit-of-RLVR are actively-developed forks;
# on_start.sh clones/pulls them onto the persistent $WORKSPACE volume at
# every boot so code is always current. Only the heavy, slow-to-install
# dependency stacks live in the image.
#
# Build (needs an x86_64 host — see docker/README.md):
#   docker build -f docker/Dockerfile -t <tag> .

ARG BASE_IMAGE=vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1

# Common consumer/datacenter archs actually rentable on Vast for this project
# (3090=8.6, 4090=8.9, A100=8.0, H100=9.0). Narrows flash-attn's build matrix
# so the CI build doesn't try to cover every arch NVIDIA has ever shipped.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# ============================================================================
# venv: es  —  ES training (Jaysen-Ma/es-at-scale @ fix/multi-engine-colocation)
# Gradient-free (population-based perturbations, vLLM-only rollouts) — no
# backward pass through transformers, so no flash-attn needed here; vLLM
# brings its own attention kernels.
# Pins mirror es-at-scale's setup.py `install_requires` exactly.
# ============================================================================
RUN uv venv /venv/es --python 3.10 && \
    . /venv/es/bin/activate && \
    uv pip install \
        "numpy" \
        "torch>=2.0.0" \
        "transformers==4.57.6" \
        "vllm==0.11.0" \
        "accelerate>=0.20.0" \
        "matplotlib>=3.5.0" \
        "psutil" \
        "datasets" \
        "ray[default]"

# ============================================================================
# venv: verl  —  GRPO training (verl-project/verl @ v0.8.0)
# Pinned to v0.8.0 (not the latest v0.9.0 tag) deliberately: v0.8.0 bounds
# vllm to the range (>=0.8.5,<=0.12.0) verl's own CI actually tests its
# FSDP<->vLLM weight-sync against; v0.9.0 declares vllm>=0.18.0 with no
# upper bound and transformers>=5.5.3,<5.11 (untested-fresh at pin time).
# Installed editable from a vendored /opt/verl copy — not $WORKSPACE, since
# verl is a framework dependency here, not one of the project's own forks.
# If you need to patch verl itself, clone your own fork onto $WORKSPACE and
# reinstall (see docker/README.md).
#
# flash-attn needs torch importable at build time, which fails under pip's
# default build isolation — install it as its own --no-build-isolation step
# AFTER verl's other deps, not via the `gpu` extras_require entry.
# ============================================================================
ARG VERL_REF=v0.8.0
RUN git clone --depth 1 --branch ${VERL_REF} https://github.com/volcengine/verl.git /opt/verl && \
    uv venv /venv/verl --python 3.10 && \
    . /venv/verl/bin/activate && \
    cd /opt/verl && \
    uv pip install -e ".[vllm,math]" && \
    MAX_JOBS=4 uv pip install --no-build-isolation flash-attn && \
    uv pip install liger-kernel

# ============================================================================
# venv: eval  —  pass@k generation + grading (limit-of-RLVR/math/examples/math_eval)
# Pins mirror limit-of-RLVR's math/requirements.txt exactly (branch
# fix/math-equal-timeout-bypass). Inlined rather than cloned at build time so
# the dependency set stays reproducible even if that branch moves; the actual
# harness code is still cloned fresh at boot by on_start.sh.
# vLLM-only inference (no backward pass) — no flash-attn needed.
# ============================================================================
RUN uv venv /venv/eval --python 3.10 && \
    . /venv/eval/bin/activate && \
    uv pip install \
        "accelerate" \
        "codetiming" \
        "datasets" \
        "dill" \
        "hydra-core==1.4.0.dev1" \
        "omegaconf==2.4.0.dev3" \
        "numpy" \
        "pybind11" \
        "ray[default]==2.10.0" \
        "tensordict" \
        "transformers<4.48" \
        "vllm<=0.6.3" \
        "peft" \
        "liger-kernel" \
        "word2number" \
        "math-verify[antlr4_11_0]==0.6.0"

# uv/pip caches purged via UV_NO_CACHE/PIP_NO_CACHE_DIR above; drop apt lists too.
RUN rm -rf /var/lib/apt/lists/*

COPY docker/on_start.sh /opt/es-capacity/on_start.sh
RUN chmod +x /opt/es-capacity/on_start.sh

LABEL org.opencontainers.image.source="https://github.com/Jaysen-Ma/ES-capacity" \
      org.opencontainers.image.description="ES training + verl GRPO training + pass@k analysis for ES-capacity"
