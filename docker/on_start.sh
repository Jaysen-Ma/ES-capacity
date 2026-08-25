#!/usr/bin/env bash
# Boot hook for the ES-capacity image. Clones/pulls the project's forks onto the
# persistent $WORKSPACE volume so code is always current without rebuilding the
# image. Idempotent: safe to run on every boot.
#
# Only dependency stacks live in the image (/venv/train, /venv/eval); the repos
# below are actively developed and must not be baked in.
set -uo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPOS="${WORKSPACE}/repos"
mkdir -p "${REPOS}"

# repo_url  target_dir  branch
sync_repo() {
    local url="$1" dir="$2" branch="$3"
    if [[ -d "${dir}/.git" ]]; then
        echo "[on_start] pulling ${dir} (${branch})"
        git -C "${dir}" fetch --quiet origin "${branch}" \
            && git -C "${dir}" checkout --quiet "${branch}" \
            && git -C "${dir}" pull --quiet --ff-only origin "${branch}" \
            || echo "[on_start] WARN: pull failed for ${dir}; leaving working copy as-is"
    else
        echo "[on_start] cloning ${url} -> ${dir} (${branch})"
        git clone --quiet --branch "${branch}" "${url}" "${dir}" \
            || echo "[on_start] WARN: clone failed for ${url}"
    fi
}

# ES_CAPACITY_REF lets an instance test a branch without a rebuild or a merge to
# main. It stays a NAMED ref on purpose: a run is only reconstructable if you can
# say which commit produced it, so "whatever is newest" is not an option here.
# Note the clone below is not --single-branch, so every branch's objects arrive
# regardless; this only selects what gets checked out.
sync_repo https://github.com/Jaysen-Ma/ES-capacity.git      "${REPOS}/ES-capacity"      "${ES_CAPACITY_REF:-main}"
sync_repo https://github.com/Jaysen-Ma/es-at-scale.git      "${REPOS}/es-at-scale"      fix/multi-engine-colocation
sync_repo https://github.com/Jaysen-Ma/limit-of-RLVR.git    "${REPOS}/limit-of-RLVR"    fix/math-equal-timeout-bypass
sync_repo https://github.com/Jaysen-Ma/simpleRL-reason.git  "${REPOS}/simpleRL-reason"  v1

# ---------------------------------------------------------------------------
# Editable installs of the two repo-vendored packages.
#
# The image ships each stack's DEPENDENCIES but not these packages themselves,
# because the repos are cloned at boot rather than baked in. Without this block
# a fresh instance fails as soon as you try to use it:
#   es_at_scale  -> `python es_at_scale/train.py` dies with ModuleNotFoundError
#   latex2sympy2 -> math_eval/grader.py dies with ModuleNotFoundError at import,
#                   i.e. the whole pass@k harness is unusable
#
# --no-deps is REQUIRED for both, not a nicety:
#   es-at-scale/setup.py lists "futures" (a Python-2 backport that cannot build
#   on 3.12) and re-pins vllm/transformers the image already satisfies;
#   latex2sympy's own pins would drag sympy/antlr backwards and break the
#   deliberately frozen /venv/eval stack.
# ---------------------------------------------------------------------------
pip_install_editable() {
    local venv="$1" pkg_dir="$2" import_name="$3"
    [[ -d "${pkg_dir}" ]] || { echo "[on_start] WARN: ${pkg_dir} missing; skipping"; return; }
    if "${venv}/bin/python" -c "import ${import_name}" >/dev/null 2>&1; then
        echo "[on_start] ${import_name} already importable in ${venv}"
        return
    fi
    echo "[on_start] installing ${pkg_dir} -> ${venv}"
    TMPDIR="${WORKSPACE}/tmp" uv pip install --python "${venv}/bin/python" \
        --no-deps -e "${pkg_dir}" \
        || echo "[on_start] WARN: install of ${pkg_dir} failed"
}

mkdir -p "${WORKSPACE}/tmp"
pip_install_editable /venv/train "${REPOS}/es-at-scale" es_at_scale
pip_install_editable /venv/eval \
    "${REPOS}/limit-of-RLVR/math/examples/math_eval/latex2sympy" latex2sympy2

# ---------------------------------------------------------------------------
# Environment for interactive shells.
#
# ES_AT_SCALE_PATH: the ES grader is imported by verl's custom reward function
# at training time. Previously this was appended to $WORKSPACE/.env, which
# NOTHING ever sourced -- it worked only because es_reward_verl.py hardcodes the
# same path as its default. Written to /etc/profile.d so it is actually loaded.
#
# HF_HOME/TMPDIR/PIP_CACHE_DIR: keep model downloads and build temp off the
# container overlay, which is small on Vast (~8GB) relative to the persistent
# volume. Filling it presents as pip "connection timeouts", not as a disk error
# -- see SETUP_NOTES.md.
# ---------------------------------------------------------------------------
cat > /etc/profile.d/es-capacity.sh <<PROFILE
export ES_AT_SCALE_PATH="${REPOS}/es-at-scale"
export HF_HOME="${WORKSPACE}/.hf_home"
export TMPDIR="${WORKSPACE}/tmp"
export PIP_CACHE_DIR="${WORKSPACE}/.pip_cache"
export UV_CACHE_DIR="${WORKSPACE}/.uv_cache"
PROFILE
chmod 0644 /etc/profile.d/es-capacity.sh
mkdir -p "${WORKSPACE}/.hf_home" "${WORKSPACE}/.pip_cache" "${WORKSPACE}/.uv_cache"

# Kept for non-login shells and tooling that reads a dotenv file directly.
grep -q 'ES_AT_SCALE_PATH' "${WORKSPACE}/.env" 2>/dev/null || {
    echo "ES_AT_SCALE_PATH=\"${REPOS}/es-at-scale\"" >> "${WORKSPACE}/.env"
}

# verl starts its own Ray head; the image's ray service would collide on :6379.
# vllm/model-ui hold GPU memory, which distorts vLLM's gpu_memory_utilization
# accounting (computed against TOTAL device memory, not free).
# Left commented deliberately — stopping services on every boot would break the
# instance's normal inference use. Uncomment, or run before a training job.
# supervisorctl stop vllm model-ui ray

# Fail loudly at boot rather than an hour into a rented-GPU session.
preflight() {
    local venv="$1" import_name="$2" label="$3"
    if "${venv}/bin/python" -c "import ${import_name}" >/dev/null 2>&1; then
        echo "  ${label}: OK"
    else
        echo "  ${label}: **BROKEN** (import ${import_name} failed)"
    fi
}

cat <<EOF
[on_start] ready.
  training + ES : source /venv/train/bin/activate
  pass@k eval   : source /venv/eval/bin/activate
  repos         : ${REPOS}
  ES-capacity   : ${ES_CAPACITY_REF:-main}
EOF
echo "[on_start] preflight:"
preflight /venv/train es_at_scale        "ES trainer      "
preflight /venv/train verl               "verl GRPO       "
preflight /venv/train flash_attn         "flash-attn      "
preflight /venv/eval  latex2sympy2       "pass@k grader   "
