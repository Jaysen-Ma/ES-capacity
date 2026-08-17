# Copy this file to config.sh (gitignored) and fill in the paths for your
# environment. scripts/run_eval.sh sources config.sh automatically if it
# exists. Nothing here is assumed by default — on a box that already
# followed docker/on_start.sh's layout, run_eval.sh falls back to
# $WORKSPACE/limit-of-RLVR/math/examples/math_eval if config.sh is absent
# and MATH_EVAL_DIR isn't otherwise set; everywhere else, set it explicitly.

# Where Jaysen-Ma/limit-of-RLVR (fix/math-equal-timeout-bypass) is checked
# out — scripts/run_eval.sh drives its math_eval harness directly.
export MATH_EVAL_DIR=/path/to/limit-of-RLVR/math/examples/math_eval
