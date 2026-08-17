#!/bin/bash
# Vast "On-start Script" for the ES-capacity training template.
# Runs on every instance boot (fresh instance AND stop/start of an existing
# one) — idempotent: clones repos onto $WORKSPACE if missing, otherwise pulls.
# Heavy deps (torch/vllm/verl/flash-attn) are already baked into the image's
# three venvs (/venv/es, /venv/verl, /venv/eval) — this script only syncs the
# actively-developed project code, which changes far more often than deps do.
set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

log() { echo "[on_start] $*"; }

sync_repo() {
    local url=$1 dir=$2 branch=$3
    if [ -d "$dir/.git" ]; then
        log "updating $dir ($branch)"
        git -C "$dir" fetch --depth 1 origin "$branch" \
            && git -C "$dir" checkout -q "$branch" \
            && git -C "$dir" reset -q --hard "origin/$branch" \
            || log "WARNING: update failed for $dir — leaving existing checkout as-is"
    else
        log "cloning $dir ($branch)"
        git clone --depth 1 --branch "$branch" "$url" "$dir" \
            || log "WARNING: clone failed for $dir"
    fi
}

sync_repo https://github.com/Jaysen-Ma/ES-capacity.git       "$WORKSPACE/ES-capacity"    main
sync_repo https://github.com/Jaysen-Ma/es-at-scale.git       "$WORKSPACE/es-at-scale"    fix/multi-engine-colocation
sync_repo https://github.com/Jaysen-Ma/limit-of-RLVR.git     "$WORKSPACE/limit-of-RLVR"  fix/math-equal-timeout-bypass

# es-at-scale and math_eval are run as scripts from inside their repo dirs
# (see ES-capacity/README.md) — no pip install needed, deps already live in
# /venv/es and /venv/eval respectively.

# verl is vendored+editable-installed into /venv/verl from /opt/verl (baked
# at image build time, pinned to v0.8.0). If you need to patch verl itself,
# clone your own fork onto $WORKSPACE and `/venv/verl/bin/pip install -e .
# --no-deps` from there instead — deps are already satisfied by the image.

# 8x RTX 3090/4090, no GPU-to-GPU P2P on these boxes (nvidia-smi topo -p2p r
# reports CNS for every pair) — required for both es-at-scale multi-engine
# vLLM and verl FSDP, or NCCL hangs at init.
ENV_FILE="$WORKSPACE/.env"
grep -q '^NCCL_P2P_DISABLE=' "$ENV_FILE" 2>/dev/null || echo 'NCCL_P2P_DISABLE=1' >> "$ENV_FILE"
export NCCL_P2P_DISABLE=1

if command -v vast-capabilities >/dev/null 2>&1; then
    IS_VOLUME=$(vast-capabilities 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["instance"]["workspace_is_volume"])' 2>/dev/null)
    if [ "$IS_VOLUME" != "True" ]; then
        log "WARNING: \$WORKSPACE is NOT a persistent volume — checkpoints/results will be lost on recycle/destroy. Sync outputs to HF Hub / off-box as they're produced."
    fi
fi

log "done. venvs: /venv/es (ES training), /venv/verl (GRPO training), /venv/eval (pass@k eval)."
log "e.g.: source /venv/es/bin/activate && cd \$WORKSPACE/es-at-scale && python -m es_at_scale.train ..."
