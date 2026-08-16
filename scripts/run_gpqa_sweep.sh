#!/bin/bash
# GPQA-diamond zero-shot across the 6 published arms (both scales x base/ES/RL).
#
# This is the out-of-domain probe, NOT pass@k: lm_eval scores the 4 choices by
# log-likelihood, one pass, no sampling. ~1.5 min/model on a single GPU, so the
# whole sweep is ~9 min.
#
# results/README.md reports 8 arms. The two extra ones (7B sigma=0.001 iter100,
# 7B sigma=0.0025 iter50) are unpublished checkpoints, so they cannot be named
# by Hub id here — add them by pointing at a local directory, e.g.
#   MODELS[7B-ES-iter100]=/path/to/experiments/qwen7b-math-run/hf-checkpoint-iter100
# and appending the key to the `for name in ...` list below.
#
# Reduce the resulting tree to the committed CSVs with:
#   python scripts/analyze_gpqa.py --sweep-root <output_dir> --out results/gpqa \
#     --pair 7B-ES:7B-base ...
#
# Usage:
#   ./run_gpqa_sweep.sh [output_dir=gpqa_results] [gpu=0]
#
# Requires: lm_eval with the vllm backend (pip install lm-eval[vllm]).
set -uo pipefail

OUT_DIR=${1:-gpqa_results}
GPU=${2:-0}

# Published checkpoints, so this runs anywhere. Swap any value for a local
# directory to score a checkpoint that hasn't been pushed to the Hub.
declare -A MODELS=(
  [1.5B-base]="Qwen/Qwen2.5-1.5B"
  [1.5B-ES]="zocrate/Qwen2.5-1.5B-ES-math"
  [1.5B-RL]="hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo"
  [7B-base]="Qwen/Qwen2.5-7B"
  [7B-ES]="zocrate/Qwen2.5-7B-ES-math"
  [7B-RL]="hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo"
)

mkdir -p "$OUT_DIR"

for name in 1.5B-base 1.5B-ES 1.5B-RL 7B-base 7B-ES 7B-RL; do
  path="${MODELS[$name]}"
  echo "########################################"
  echo "# GPQA-diamond: $name  ($path)"
  echo "########################################"
  date
  CUDA_VISIBLE_DEVICES="$GPU" lm_eval --model vllm \
    --model_args pretrained="$path",dtype=bfloat16,gpu_memory_utilization=0.85 \
    --tasks gpqa_diamond_zeroshot \
    --batch_size auto \
    --output_path "$OUT_DIR/$name" \
    --log_samples
  date
  echo "=== Done: $name ==="
  echo
done

echo "ALL GPQA RUNS COMPLETE"
