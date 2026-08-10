#!/bin/bash
# End-to-end: generate the third model's outputs on all 4 benchmarks, then
# produce the 3-way plot + dual four-way tables for each, reusing the base
# and ES-trained outputs already computed by run_full_eval_suite.sh.
#
# Usage:
#   ./run_third_model_full_suite.sh <model_dir> <label> <run_tag>
# e.g.
#   ./run_third_model_full_suite.sh <hf-snapshot-dir> rl iter50
set -euo pipefail

MODEL_DIR=$1
LABEL=$2
RUN_TAG=${3:-run1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="/workspace/ES-capacity/results/${RUN_TAG}"

"$SCRIPT_DIR/run_third_model_eval.sh" "$MODEL_DIR" "$LABEL" "$RUN_TAG"

declare -A TITLES=(
    [aime24]="AIME24"
    [math500]="MATH500"
    [minerva_math]="Minerva"
    [olympiadbench]="OlympiadBench"
)

cd /workspace/ES-capacity
for benchmark in aime24 math500 minerva_math olympiadbench; do
    echo
    echo "=== Three-way analysis: $benchmark ==="
    python scripts/analyze_three_way.py \
        --base-dir "${OUT_ROOT}/base/${benchmark}" \
        --es-dir "${OUT_ROOT}/trained/${benchmark}" \
        --rl-dir "${OUT_ROOT}/${LABEL}/${benchmark}" \
        --out-prefix "${OUT_ROOT}/${benchmark}_threeway" \
        --title "${TITLES[$benchmark]}" \
        --plot
done

echo
echo "All three-way analyses complete. See results/${RUN_TAG}/{aime24,math500,minerva_math,olympiadbench}_threeway_summary.json"
