#!/usr/bin/env bash
# Evaluate one EGGROLL checkpoint against the completed base/GRPO v1 arms and
# regenerate the three-arm figures. Reusable for the post-block eval and for
# any later on-demand check of an intermediate checkpoint.
#
# Usage: eval_eggroll_checkpoint.sh <checkpoint_dir> <label> [profile]
#   checkpoint_dir  absolute or repo-relative path to a merged HF checkpoint
#   label           short tag, becomes run id v1_eggroll_7b_<label> and
#                   figures dir runs/figures/v1_three_arms_<label>
#   profile         eval profile (default: v1, full Minerva n=64). Use n8 for
#                   a quick ~15min look via configs/experiments/minerva_n8.toml
#                   instead (pass "n8" here).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source "$SCRIPT_DIR/_env.sh"
MACHINE="${ES_CAPACITY_MACHINE:-gb10}"

CKPT="$1"
LABEL="$2"
PROFILE="${3:-v1}"
RUN_ID="v1_eggroll_7b_${LABEL}"
FIG_DIR="runs/figures/v1_three_arms_${LABEL}"

echo "[$(date -Is)] START eval ckpt=$CKPT label=$LABEL profile=$PROFILE run_id=$RUN_ID"
python - "$MACHINE" "$PROFILE" "$CKPT" "$RUN_ID" <<'PY'
import sys
from pathlib import Path
from es_capacity.config import load_config
from es_capacity.generate import run_eval_shards

machine, profile, ckpt, run_id = sys.argv[1:5]
cfg = load_config(machine=machine, profile=profile)
cfg.raw.setdefault("models", {})["eggroll"] = {"path": str(Path(ckpt).resolve()), "hf_id": ""}
run_dir = run_eval_shards(cfg, arm="eggroll", model_key="eggroll", profile=profile, run_id=run_id)
print(f"eggroll eval done: {run_dir}")
PY
echo "[$(date -Is)] DONE eval"

echo "[$(date -Is)] START figures"
python -m es_capacity.cli.figures \
  --machine "$MACHINE" \
  --runs base=v1_base_7b_preexp,grpo=v1_grpo_7b,eggroll="$RUN_ID" \
  --out-dir "$FIG_DIR" \
  --base-key base
echo "[$(date -Is)] EVAL_CHAIN_COMPLETE label=$LABEL fig_dir=$FIG_DIR"
