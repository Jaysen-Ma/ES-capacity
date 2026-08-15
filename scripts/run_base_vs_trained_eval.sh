#!/bin/bash
# Runs the base model and the ES-trained checkpoint through IDENTICAL
# generation settings, so pass@k / solvable-fraction comparisons are fair.
# Both invocations share every flag (template, temperature, top_p, max_tokens,
# n_sampling, seed, GPU sharding) except the model path — that's the only
# thing this script is trying to guarantee, so don't override just one side's
# args when calling this.
#
# Usage:
#   ./run_base_vs_trained_eval.sh <trained_hf_model_dir> <run_tag> [benchmark=minerva_math] [n_sampling=64] [base_model_dir]
#
# base_model_dir defaults to Experiment 1's Qwen2.5-1.5B snapshot; pass it
# explicitly to compare against a different base (e.g. Qwen2.5-7B).
#
# Requires: limit-of-RLVR/math/examples/math_eval/run_sharded_eval.sh
set -euo pipefail

TRAINED_MODEL_DIR=$1
RUN_TAG=${2:-run1}
BENCHMARK=${3:-minerva_math}
N_SAMPLING=${4:-64}

BASE_MODEL_DIR=${5:-"/workspace/.hf_home/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"}
PROMPT_TYPE="qwen-boxed"
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=2048
SEED=1

MATH_EVAL_DIR="/workspace/limit-of-RLVR/math/examples/math_eval"
OUT_ROOT="/workspace/ES-capacity/results/${RUN_TAG}"
BASE_OUT="${OUT_ROOT}/base"
TRAINED_OUT="${OUT_ROOT}/trained"

mkdir -p "$BASE_OUT" "$TRAINED_OUT"

echo "=== Fixed eval settings (identical for both models) ==="
echo "benchmark=$BENCHMARK prompt_type=$PROMPT_TYPE n_sampling=$N_SAMPLING temperature=$TEMPERATURE top_p=$TOP_P max_tokens=$MAX_TOKENS seed=$SEED"
echo

cd "$MATH_EVAL_DIR"

echo "=== [1/2] Base model: $BASE_MODEL_DIR ==="
./run_sharded_eval.sh "$BASE_MODEL_DIR" "$BASE_OUT" "$BENCHMARK" "$N_SAMPLING" \
    "$TEMPERATURE" "$TOP_P" "$MAX_TOKENS" "$PROMPT_TYPE" "$SEED"

echo "=== [2/2] Trained model: $TRAINED_MODEL_DIR ==="
./run_sharded_eval.sh "$TRAINED_MODEL_DIR" "$TRAINED_OUT" "$BENCHMARK" "$N_SAMPLING" \
    "$TEMPERATURE" "$TOP_P" "$MAX_TOKENS" "$PROMPT_TYPE" "$SEED"

case "$BENCHMARK" in
    aime24) TITLE="AIME24" ;;
    math500) TITLE="MATH500" ;;
    minerva_math) TITLE="Minerva" ;;
    olympiadbench) TITLE="OlympiadBench" ;;
    *) TITLE="$BENCHMARK" ;;
esac

echo "=== Analyzing ==="
cd /workspace/ES-capacity
python scripts/analyze_results.py \
    --base-dir "${BASE_OUT}/${BENCHMARK}" \
    --trained-dir "${TRAINED_OUT}/${BENCHMARK}" \
    --out-prefix "results/${RUN_TAG}/${BENCHMARK}" \
    --title "$TITLE" \
    --plot

echo "Done. See results/${RUN_TAG}/${BENCHMARK}_summary.json and _passk.png"
