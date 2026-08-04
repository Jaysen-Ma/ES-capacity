#!/usr/bin/env bash
# Wait for v1 pipeline to finish, then train EGGROLL 7B and evaluate it.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source "$SCRIPT_DIR/_env.sh"
PIPE_LOG=logs/v1_pipeline.nohup.out

echo "[$(date -Is)] waiting for v1 pipeline..."
while ! grep -q 'PIPELINE COMPLETE' "$PIPE_LOG" 2>/dev/null; do
  sleep 60
done
echo "[$(date -Is)] pipeline done; starting EGGROLL 7B train"

bash scripts/train_eggroll_7b.sh

# Evaluate checkpoint
export EGG_CKPT=runs/train/eggroll_7b_hf/checkpoint
python - <<'PY'
import os
from es_capacity.config import load_config
from es_capacity.generate import run_eval_shards
from es_capacity.figures import make_all_figures
from pathlib import Path
cfg = load_config(machine=os.environ.get("ES_CAPACITY_MACHINE", "gb10"), profile="v1")
ckpt = Path("runs/train/eggroll_7b_hf/checkpoint").resolve()
cfg.raw.setdefault("models", {})["eggroll"] = {"path": str(ckpt), "hf_id": ""}
run_eval_shards(cfg, arm="eggroll", model_key="eggroll", profile="v1", run_id="v1_eggroll_7b")
make_all_figures(
    {
        "base": Path("runs/v1_base_7b"),
        "grpo": Path("runs/v1_grpo_7b"),
        "eggroll": Path("runs/v1_eggroll_7b"),
    },
    Path("runs/figures/v1_three_arms"),
    base_key="base",
)
print("three-arm figures done")
PY
echo "[$(date -Is)] EGGROLL follow-up COMPLETE"
