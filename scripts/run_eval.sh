#!/bin/bash
# Runs ONE model across all 4 benchmarks with the fixed generation settings
# every arm in this repo shares (temperature=0.6, top_p=0.95, max_tokens=2048,
# qwen-boxed template, seed=1), so pass@k comparisons between arms stay fair.
# Call this once per arm (base, trained, rl, ...) under the same run_tag, then
# analyze_passk.py compares whichever labels you point it at.
#
# Usage:
#   ./run_eval.sh <model_dir> <label> <run_tag> [n_sampling_override]
# e.g.
#   ./run_eval.sh /path/to/hf/Qwen2.5-7B-snapshot base 7b-sigma001-iter50
#   ./run_eval.sh /path/to/es-at-scale/experiments/qwen7b-math-run/hf-checkpoint-iter50 trained 7b-sigma001-iter50
#   ./run_eval.sh /path/to/hf/SimpleRL-Zoo-snapshot rl 7b-sigma001-iter50
#   ./run_eval.sh /path/to/hf/SimpleRL-Zoo-snapshot rl 7b-sigma001-iter50 32   # same n_sampling=32 for all 4 benchmarks
#
# n_sampling_override, if given, replaces every benchmark's n_sampling below
# (e.g. for a cheap "does it already cross at low k" check before committing
# to the full budget). Without it, this model gets this project's standard
# per-benchmark n_sampling (harder/smaller benchmarks get a larger k), so
# arms evaluated separately under the same run_tag stay directly comparable.
#
# Needs MATH_EVAL_DIR pointing at a limit-of-RLVR checkout's
# math/examples/math_eval/ (this repo doesn't vendor it, and its path isn't
# assumed — copy config.example.sh to config.sh and set it there, or export
# it yourself; on a box following docker/on_start.sh's layout it defaults to
# $WORKSPACE/limit-of-RLVR/math/examples/math_eval).
set -euo pipefail

MODEL_DIR=$1
LABEL=$2
RUN_TAG=${3:-run1}
N_SAMPLING_OVERRIDE=${4:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$REPO_ROOT/config.sh" ]; then
    source "$REPO_ROOT/config.sh"
fi
if [ -z "${MATH_EVAL_DIR:-}" ] && [ -n "${WORKSPACE:-}" ]; then
    MATH_EVAL_DIR="$WORKSPACE/limit-of-RLVR/math/examples/math_eval"
fi
if [ -z "${MATH_EVAL_DIR:-}" ]; then
    echo "ERROR: MATH_EVAL_DIR is not set." >&2
    echo "       Copy config.example.sh to config.sh and set it there, or:" >&2
    echo "       export MATH_EVAL_DIR=/path/to/limit-of-RLVR/math/examples/math_eval" >&2
    exit 1
fi

PROMPT_TYPE="qwen-boxed"
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=2048
SEED=1

# benchmark:n_sampling pairs
PAIRS=(
    "aime24:512"
    "math500:128"
    "minerva_math:128"
    "olympiadbench:128"
)

OUT_ROOT="${REPO_ROOT}/results/${RUN_TAG}"
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
echo "Analyze with: python ${SCRIPT_DIR}/analyze_passk.py --run-tag ${RUN_TAG} --label ... --baseline ... --plot"
