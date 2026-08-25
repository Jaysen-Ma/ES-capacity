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
# ABI NOTE: the wheel's cxx11abi tag must match torch's. The rule is VERSION-
# based, not index-based: every torch >= 2.7 linux wheel is built with
# _GLIBCXX_USE_CXX11_ABI=1 after PyTorch's manylinux_2_28 migration, and the
# default PyPI wheel for 2.8.0 is itself the cu128 build. vllm 0.11.0 pulls
# torch 2.8.0+cu128 from stock PyPI here -- no index-url is set anywhere in
# this repo -- so abiTRUE is correct. There is no ABI=0 torch 2.8 wheel to
# reason from; an earlier revision assumed one and this assert caught it.
# Assert, don't assume: a flip fails here, not at `import flash_attn` on a
# rented GPU box.
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
ARG FA_ABI=TRUE

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
#
# math-verify is PINNED here on purpose. It arrives transitively from verl's
# "math" extra (verl/setup.py: MATH_REQUIRES = ["math-verify"], unversioned), so
# without this line the version floats with the build date -- and it is not an
# incidental dependency: es_at_scale's boxed_reward_fn calls it (via
# is_latex_equal) as the final arm of its correctness check, so the version
# silently redefines the TRAINING REWARD both arms optimize. Measured drift:
# 0.6.0 grades `\boxed{50\%}` against `0.5` as correct, 0.9.0 does not.
#
# NOTE: /venv/eval is frozen at math-verify 0.6.0, so the two venvs disagree by
# construction. That is a deliberate open question, not an oversight -- see
# docker/README.md "Grader versioning". Do not "fix" it by aligning the numbers
# without deciding which grader the published results should be attributed to.
#
# pylatexenc is listed explicitly: math_grader.py imports it directly, and
# nothing else in the stack pulls it in.
#
# --seed gives /venv/train a real pip. uv venv does not seed one, so every
# `pip install ...` instruction -- including es-at-scale's own README -- fails
# with "No module named pip" on a stack that is otherwise fine.
RUN . /venv/train/bin/activate \
 && uv pip install numpy accelerate "matplotlib>=3.5.0" psutil datasets "ray[default]" \
    "math-verify==0.9.0" "pylatexenc==2.11" \
 && uv pip install --python /venv/train/bin/python pip \
 && python -c "import math_verify, pylatexenc; print('graders OK')"

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
#
# vllm is pinned ==0.6.3, NOT <=0.6.3. An upper bound with no floor is not a
# freeze: the resolver walked it down to vllm 0.5.0.post1 / torch 2.3.0, and
# would drift again on a different build date -- silently breaking the very
# comparability this venv exists to preserve.
#
# setuptools is listed explicitly because `uv venv` does not seed it and vllm
# imports it at runtime. Without it the venv builds clean and then dies at
# `import vllm` -- i.e. at eval time, not build time.
# ============================================================================
RUN uv venv /venv/eval --python 3.10 \
 && . /venv/eval/bin/activate \
 && uv pip install --prerelease=allow \
    accelerate codetiming datasets dill \
    "hydra-core==1.4.0.dev1" "omegaconf==2.4.0.dev3" \
    numpy pybind11 "ray[default]==2.10.0" setuptools tensordict \
    "transformers<4.48" "vllm==0.6.3" peft liger-kernel word2number \
    "math-verify[antlr4_11_0]==0.6.0"

# Repos are NOT baked in — ES-capacity, es-at-scale, limit-of-RLVR and
# simpleRL-reason are actively-developed forks. on_start.sh clones/pulls them
# onto the persistent $WORKSPACE volume at every boot, so code is always
# current without rebuilding a 30GB image.
COPY docker/on_start.sh /opt/es-capacity/on_start.sh
# Strip CR before chmod. A Windows checkout (core.autocrlf=true) yields a CRLF
# on_start.sh; Linux then rejects it with `bad interpreter: bash^M` and the
# instance boots with no repos cloned. .gitattributes fixes new clones, but the
# image must be correct regardless of the builder's git config.
RUN sed -i 's/\r//g' /opt/es-capacity/on_start.sh \
 && chmod +x /opt/es-capacity/on_start.sh

RUN rm -rf /var/lib/apt/lists/* /root/.cache /tmp/*

# Runtime-only environment. Deliberately placed AFTER every RUN step: these
# point at $WORKSPACE, which does not exist at build time, so setting them
# earlier would redirect the build's own temp/cache into a nonexistent path.
#
# On Vast the container overlay is ~8GB against a much larger persistent volume,
# and overflowing it surfaces as pip "connection timeout" errors rather than a
# disk error -- hours of misdiagnosis (see SETUP_NOTES.md). on_start.sh creates
# these directories and re-exports the same values via /etc/profile.d.
# PIP_NO_CACHE_DIR/UV_NO_CACHE are set at the top to keep the BUILD small; they
# are re-enabled here so the *_CACHE_DIR values below are not dead letters at
# runtime, where caching across sessions is what you actually want.
ENV HF_HOME=/workspace/.hf_home \
    TMPDIR=/workspace/tmp \
    PIP_CACHE_DIR=/workspace/.pip_cache \
    UV_CACHE_DIR=/workspace/.uv_cache \
    PIP_NO_CACHE_DIR=0 \
    UV_NO_CACHE=0

LABEL org.opencontainers.image.source="https://github.com/Jaysen-Ma/ES-capacity" \
      org.opencontainers.image.description="ES + GRPO training and pass@k eval for ES-capacity"
