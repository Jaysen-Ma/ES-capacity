#!/usr/bin/env bash
# Sequential Minerva v1 eval pipeline: base → grpo → (optional) eggroll checkpoint
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source "$SCRIPT_DIR/_env.sh"
MACHINE="${ES_CAPACITY_MACHINE:-gb10}"
mkdir -p logs

echo "[$(date -Is)] START v1 base"
python -m es_capacity.cli.eval --machine "$MACHINE" --profile v1 --arm base --model-key base --run-id v1_base_7b \
  2>&1 | tee logs/v1_base_7b.log
echo "[$(date -Is)] DONE v1 base"

echo "[$(date -Is)] START v1 grpo"
python -m es_capacity.cli.eval --machine "$MACHINE" --profile v1 --arm grpo --model-key grpo --run-id v1_grpo_7b \
  2>&1 | tee logs/v1_grpo_7b.log
echo "[$(date -Is)] DONE v1 grpo"

# Figures for two arms
python -m es_capacity.cli.figures \
  --machine "$MACHINE" \
  --runs base=v1_base_7b,grpo=v1_grpo_7b \
  --out-dir runs/figures/v1_base_vs_grpo \
  --base-key base
echo "[$(date -Is)] wrote two-arm figures"

# If EGGROLL checkpoint exists, evaluate it
EGG_CKPT="${EGG_CKPT:-runs/train/eggroll_7b_hf/checkpoint}"
if [[ -d "$EGG_CKPT" ]]; then
  echo "[$(date -Is)] START v1 eggroll from $EGG_CKPT"
  # Temporarily point a model key via symlink under models dir is awkward;
  # use absolute path override by writing a one-off local override.
  python - <<PY
from es_capacity.config import load_config
from es_capacity.generate import run_eval_shards
from pathlib import Path
cfg = load_config(machine="$MACHINE", profile="v1")
# Inject eggroll path
cfg.raw.setdefault("models", {})["eggroll"] = {"path": str(Path("$EGG_CKPT").resolve()), "hf_id": ""}
run_eval_shards(cfg, arm="eggroll", model_key="eggroll", profile="v1", run_id="v1_eggroll_7b")
print("eggroll eval done")
PY
  python -m es_capacity.cli.figures \
    --machine "$MACHINE" \
    --runs base=v1_base_7b,grpo=v1_grpo_7b,eggroll=v1_eggroll_7b \
    --out-dir runs/figures/v1_three_arms \
    --base-key base
fi

echo "[$(date -Is)] PIPELINE COMPLETE"
