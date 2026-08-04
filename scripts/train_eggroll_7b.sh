#!/usr/bin/env bash
# Train EGGROLL on the base 7B model (HF in-place low-rank). Gate: prior 1.5B validate + Multi-LoRA probe.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source "$SCRIPT_DIR/_env.sh"
MACHINE="${ES_CAPACITY_MACHINE:-gb10}"

PROBE=runs/probes/multilora_sm121.json
if [[ ! -f "$PROBE" ]] || ! grep -q '"ok": true' "$PROBE"; then
  echo "Multi-LoRA probe not green; abort"
  exit 1
fi

# Prefer base 7B to match Yue arms; Instruct used only for easy-synthetic sanity.
MODEL="${MODEL:-$(python3 -m es_capacity.config --machine "$MACHINE" --model-key base)}"
OUT="${OUT:-runs/train/eggroll_7b_hf}"

echo "[$(date -Is)] EGGROLL 7B train model=$MODEL out=$OUT"
# Longer run: pop 8, 40 iters, 8 Minerva problems, 256 tokens — still a scaled-down budget.
python -m es_capacity.posttrain.eggroll_hf_train \
  --model "$MODEL" \
  --out-dir "$OUT" \
  --population-size 8 \
  --num-iterations 40 \
  --num-problems 8 \
  --max-new-tokens 256 \
  2>&1 | tee logs/eggroll_7b_hf_train.log

echo "[$(date -Is)] EGGROLL 7B train finished"
cat "$OUT/summary.json"
