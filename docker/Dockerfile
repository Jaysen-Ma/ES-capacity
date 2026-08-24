# ES-capacity image: ES training (es-at-scale) + verl GRPO training + pass@k eval
# (limit-of-RLVR/math_eval), plus the claude and gh CLIs.
#
# Build (x86_64 host required — see docker/README.md):
#   docker build -f docker/Dockerfile -t ghcr.io/jaysen-ma/es-capacity:latest .
#
# ---------------------------------------------------------------------------
# WHY TWO VENVS, NOT THREE
#
# A previous version of this file used three (es / verl / eval) on the premise
# that all three stacks conflict. They do not. Verified against each repo's
# own metadata:
#   es-at-scale  setup.py     -> transformers==4.57.6, vllm==0.11.0
#   verl v0.7.1  setup.py:41  -> "transformers"          (UNPINNED)
#   verl v0.7.1  setup.py:52  -> vllm>=0.8.5,<=0.12.0    (0.11.0 is in range)
# so ES training and verl GRPO share /venv/train. Only the eval harness is
# genuinely incompatible (vllm<=0.6.3, transformers<4.48 -> torch 2.4).
# This halves the image and is what makes it fit a CI runner's disk.
#
# WHY FLASH-ATTN IS NOT COMPILED HERE
#
# The previous build died on GitHub Actions with "the hosted runner lost
# communication with the server" — the OOM/disk-exhaustion signature. Cause was
# `MAX_JOBS=4 pip install --no-build-isolation flash-attn` with
# TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0": a multi-hour, many-GB-RAM compile
# across four archs on a 7GB runner. Dao-AILab ships prebuilt wheels; we use
# one. No nvcc runs during this build at all.
#
# ABI NOTE: the wheel's cxx11abi tag must match torch's. torch 2.8 PyPI wheels
# are built with _GLIBCXX_USE_CXX11_ABI=0 -> the abiFALSE wheel. The build
# asserts this rather than assuming it.
# ---------------------------------------------------------------------------

# Tag verified to exist on Docker Hub. Note ubuntu24.04 (not 22.04) and the
# mandatory date suffix — vastai publishes no undated variant of this tag.
# The previous Dockerfile referenced cuda-12.8.1-...-ubuntu22.04-py310, which
# has never existed; that alone would fail the build on the FROM line.
ARG BASE_IMAGE=vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312-2026-08-21
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_CONCURRENT_DOWNLOADS=32 \
    UV_HTTP_TIMEOUT=180

# files.pythonhosted.org throttles per-connection (~425 KB/s single vs ~3.3 MB/s
# across four on the instance this was developed on). UV_CONCURRENT_DOWNLOADS
# above is a ~6x wall-clock difference on this image; do not remove it.

# ============================================================================
# CLI tooling: gh + claude
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git git-lfs jq less \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Claude Code needs node. The base image ships nvm, but nvm only lands on PATH
# in an interactive login shell — a docker RUN, a script, and a supervisor
# service all miss it. Install node globally instead so `claude` works from any
# non-interactive context.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# /venv/train — es-at-scale (ES) AND verl v0.7.1 (GRPO), sharing one stack.
#
# vllm 0.11.0 is pinned to match the ES arm's inference engine exactly. The
# ES-capacity wall-clock comparison is between ES and GRPO on the same box; if
# the two arms ran different vLLM versions it would be measuring vLLM, not the
# algorithms. See docs/matched-experiment.md.
# ============================================================================
ARG VERL_REF=v0.7.1
ARG FA_VERSION=2.8.3.post1
ARG FA_ABI=FALSE

RUN uv venv /venv/train --python 3.12 \
 && . /venv/train/bin/activate \
 && uv pip install "vllm==0.11.0" "transformers==4.57.6" \
 && git clone --depth 1 --branch ${VERL_REF} https://github.com/volcengine/verl.git /opt/verl \
 && cd /opt/verl && uv pip install -e ".[vllm,math]" \
 && uv pip install liger-kernel \
 && python -c "import torch,sys; \
    want=(sys.argv[1]=='TRUE'); got=torch._C._GLIBCXX_USE_CXX11_ABI; \
    print(f'torch {torch.__version__} cxx11abi={got}, wheel expects {want}'); \
    sys.exit(0 if got==want else 1)" ${FA_ABI} \
 && uv pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VERSION}/flash_attn-${FA_VERSION}+cu12torch2.8cxx11abi${FA_ABI}-cp312-cp312-linux_x86_64.whl" \
 && python -c "import flash_attn; print('flash-attn', flash_attn.__version__, 'OK')"

# es-at-scale's own deps on top of the shared stack. Its setup.py pins
# transformers==4.57.6 and vllm==0.11.0, already satisfied above.
RUN . /venv/train/bin/activate \
 && uv pip install numpy accelerate "matplotlib>=3.5.0" psutil datasets "ray[default]"

# ============================================================================
# /venv/eval — limit-of-RLVR math_eval (pass@k generation + grading).
#
# Pins mirror limit-of-RLVR's math/requirements.txt on branch
# fix/math-equal-timeout-bypass. Kept frozen deliberately: the published
# ES-capacity pass@k numbers were produced under this stack, so changing it
# would break comparability with results already in the repo.
#
# --prerelease=allow is REQUIRED: hydra-core 1.4.0.dev1 and omegaconf 2.4.0.dev3
# are dev releases and uv refuses them by default. This is one of the reasons
# the previous Dockerfile could not build.
# ============================================================================
RUN uv venv /venv/eval --python 3.10 \
 && . /venv/eval/bin/activate \
 && uv pip install --prerelease=allow \
    accelerate codetiming datasets dill \
    "hydra-core==1.4.0.dev1" "omegaconf==2.4.0.dev3" \
    numpy pybind11 "ray[default]==2.10.0" tensordict \
    "transformers<4.48" "vllm<=0.6.3" peft liger-kernel word2number \
    "math-verify[antlr4_11_0]==0.6.0"

# Repos are NOT baked in — ES-capacity, es-at-scale, limit-of-RLVR and
# simpleRL-reason are actively-developed forks. on_start.sh clones/pulls them
# onto the persistent $WORKSPACE volume at every boot, so code is always
# current without rebuilding a 30GB image.
COPY docker/on_start.sh /opt/es-capacity/on_start.sh
RUN chmod +x /opt/es-capacity/on_start.sh

RUN rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

LABEL org.opencontainers.image.source="https://github.com/Jaysen-Ma/ES-capacity" \
      org.opencontainers.image.description="ES + GRPO training and pass@k eval for ES-capacity"
