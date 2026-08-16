#!/bin/bash
# Runs ONE additional model (e.g. an RL-trained checkpoint like SimpleRL-Zoo)
# across all 4 benchmarks, reusing the base model's outputs already computed
# by run_full_eval_suite.sh under the same run_tag — no need to regenerate
# the base model's generations a second time.
#
# Generation settings (temperature/top_p/max_tokens/prompt_type/seed) are
# fixed identically to run_base_vs_trained_eval.sh's, so this stays a fair
# comparison against both the base model and the ES-trained arm.
#
# Usage:
#   ./run_third_model_eval.sh <model_dir> <label> <run_tag> [n_sampling_override]
# e.g.
#   ./run_third_model_eval.sh /workspace/.hf_home/.../SimpleRL-Zoo/snapshots/<hash> rl 1.5b-sigma001-iter50
#   ./run_third_model_eval.sh <hash> rl 1.5b-sigma001-iter50 32   # same n_sampling=32 for all 4 benchmarks
#
# n_sampling_override, if given, replaces every benchmark's n_sampling below
# (e.g. for a cheap "does it already cross at low k" check before committing
# to the full base/ES-matching budget). Without it, this model gets the same
# per-benchmark n_sampling as the base/ES runs, so all three stay directly
# comparable — WITH it, this model's pass@k curve will only extend to the
# override value while base/ES still have their full curves (analyze_three_way.py
# handles the mismatched k-ranges: each series is plotted over its own range).
set -euo pipefail

MODEL_DIR=$1
LABEL=$2
RUN_TAG=${3:-run1}
N_SAMPLING_OVERRIDE=${4:-}

PROMPT_TYPE="qwen-boxed"
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=2048
SEED=1

# benchmark:n_sampling pairs — must match run_full_eval_suite.sh's so the
# base-model outputs line up exactly (same n_sampling per benchmark), unless
# N_SAMPLING_OVERRIDE is set.
PAIRS=(
    "aime24:512"
    "math500:128"
    "minerva_math:128"
    "olympiadbench:128"
)

MATH_EVAL_DIR="/workspace/limit-of-RLVR/math/examples/math_eval"
OUT_ROOT="/workspace/ES-capacity/results/${RUN_TAG}"
LABEL_OUT="${OUT_ROOT}/${LABEL}"
mkdir -p "$LABEL_OUT"

cd "$MATH_EVAL_DIR"

for pair in "${PAIRS[@]}"; do
    benchmark="${pair%%:*}"
    n_sampling="${pair##*:}"
    if [ -n "$N_SAMPLING_OVERRIDE" ]; then
        n_sampling="$N_SAMPLING_OVERRIDE"
    fi
    echo
    echo "########################################"
    echo "# Benchmark: $benchmark  (n_sampling=$n_sampling, model=$LABEL)"
    echo "########################################"
    ./run_sharded_eval.sh "$MODEL_DIR" "$LABEL_OUT" "$benchmark" "$n_sampling" \
        "$TEMPERATURE" "$TOP_P" "$MAX_TOKENS" "$PROMPT_TYPE" "$SEED"
done

echo
echo "All benchmarks complete for $LABEL. Outputs in ${LABEL_OUT}/{aime24,math500,minerva_math,olympiadbench}/"
