#!/bin/bash
# Runs base-vs-trained pass@k eval across all 4 benchmarks, each using the
# full 8-GPU shard (one benchmark at a time, not split across benchmarks, to
# keep max throughput per benchmark). Reuses run_base_vs_trained_eval.sh so
# every benchmark still gets identical base/trained generation settings
# except n_sampling (which varies by benchmark per this project's convention:
# harder/smaller benchmarks get a larger k).
#
# Usage:
#   ./run_full_eval_suite.sh <trained_hf_model_dir> <run_tag> [base_model_dir]
#
# base_model_dir defaults (in run_base_vs_trained_eval.sh) to Experiment 1's
# Qwen2.5-1.5B snapshot; pass it explicitly to compare against a different base.
set -euo pipefail

TRAINED_MODEL_DIR=$1
RUN_TAG=${2:-run1}
BASE_MODEL_DIR=${3:-}

# benchmark:n_sampling pairs
PAIRS=(
    "aime24:512"
    "math500:128"
    "minerva_math:128"
    "olympiadbench:128"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for pair in "${PAIRS[@]}"; do
    benchmark="${pair%%:*}"
    n_sampling="${pair##*:}"
    echo
    echo "########################################"
    echo "# Benchmark: $benchmark  (n_sampling=$n_sampling)"
    echo "########################################"
    "$SCRIPT_DIR/run_base_vs_trained_eval.sh" "$TRAINED_MODEL_DIR" "$RUN_TAG" "$benchmark" "$n_sampling" "$BASE_MODEL_DIR"
done

echo
echo "All benchmarks complete. Summaries in results/${RUN_TAG}/{aime24,math500,minerva_math,olympiadbench}_summary.json"
