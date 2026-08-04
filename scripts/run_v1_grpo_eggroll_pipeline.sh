#!/usr/bin/env bash
# v1 eval without re-running base: import preexp base, eval GRPO, then EGGROLL train+eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source "$SCRIPT_DIR/_env.sh"
MACHINE="${ES_CAPACITY_MACHINE:-gb10}"
mkdir -p logs

echo "[$(date -Is)] Import preexp base → runs/v1_base_7b_preexp"
if [[ -n "${PREEXP_DIR:-}" ]]; then
  # Author-only shortcut: carry over completions.jsonl + metrics.json from an
  # earlier ad-hoc run instead of re-evaluating. Not needed on a fresh clone.
  python -m es_capacity.cli.import_preexp --preexp-dir "$PREEXP_DIR" --out-run runs/v1_base_7b_preexp
else
  echo "[$(date -Is)] \$PREEXP_DIR not set; running a fresh base eval instead of importing"
  python -m es_capacity.cli.eval --machine "$MACHINE" --profile v1 --arm base --model-key base \
    --run-id v1_base_7b_preexp 2>&1 | tee logs/v1_base_7b_preexp.log
fi

echo "[$(date -Is)] START v1 grpo (SimpleRL-Zoo)"
python -m es_capacity.cli.eval --machine "$MACHINE" --profile v1 --arm grpo --model-key grpo --run-id v1_grpo_7b \
  2>&1 | tee logs/v1_grpo_7b.log
echo "[$(date -Is)] DONE v1 grpo"

python -m es_capacity.cli.figures \
  --machine "$MACHINE" \
  --runs base=v1_base_7b_preexp,grpo=v1_grpo_7b \
  --out-dir runs/figures/v1_base_vs_grpo \
  --base-key base
echo "[$(date -Is)] wrote two-arm figures (base preexp + grpo)"

echo "[$(date -Is)] START EGGROLL 7B HF train"
bash scripts/train_eggroll_7b.sh

EGG_CKPT=runs/train/eggroll_7b_hf/checkpoint
if [[ -d "$EGG_CKPT" ]]; then
  echo "[$(date -Is)] START v1 eggroll eval"
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
        "base": Path("runs/v1_base_7b_preexp"),
        "grpo": Path("runs/v1_grpo_7b"),
        "eggroll": Path("runs/v1_eggroll_7b"),
    },
    Path("runs/figures/v1_three_arms"),
    base_key="base",
)
print("three-arm figures done")
PY
fi

echo "[$(date -Is)] PIPELINE COMPLETE (base=preexp, grpo+eggroll evaluated)"
