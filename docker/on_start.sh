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

# The ES grader is imported by verl's custom reward function at training time.
grep -q 'ES_AT_SCALE_PATH' "${WORKSPACE}/.env" 2>/dev/null || {
    echo "ES_AT_SCALE_PATH=\"${REPOS}/es-at-scale\"" >> "${WORKSPACE}/.env"
}

# verl starts its own Ray head; the image's ray service would collide on :6379.
# vllm/model-ui hold GPU memory, which distorts vLLM's gpu_memory_utilization
# accounting (computed against TOTAL device memory, not free).
# Left commented deliberately — stopping services on every boot would break the
# instance's normal inference use. Uncomment, or run before a training job.
# supervisorctl stop vllm model-ui ray

cat <<EOF
[on_start] ready.
  training + ES : source /venv/train/bin/activate
  pass@k eval   : source /venv/eval/bin/activate
  repos         : ${REPOS}
  ES-capacity   : ${ES_CAPACITY_REF:-main}
EOF
