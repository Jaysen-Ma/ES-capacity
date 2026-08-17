# Copy this file to config.sh (gitignored) and fill in the paths for your
# environment. scripts/run_eval.sh sources config.sh automatically if it
# exists. Nothing here is assumed by default — on a box that already
# followed docker/on_start.sh's layout, run_eval.sh falls back to
# $WORKSPACE/limit-of-RLVR/math/examples/math_eval if config.sh is absent
# and MATH_EVAL_DIR isn't otherwise set; everywhere else, set it explicitly.

# Where Jaysen-Ma/limit-of-RLVR (fix/math-equal-timeout-bypass) is checked
# out — scripts/run_eval.sh drives its math_eval harness directly.
export MATH_EVAL_DIR=/path/to/limit-of-RLVR/math/examples/math_eval

# Root directory where you keep local model snapshots/checkpoints (base
# models, ES/RL checkpoints, whatever you point run_eval.sh at). Lets you
# pass a short relative name, e.g. `run_eval.sh Qwen2.5-7B base my-run`,
# instead of a full path — resolved as $MODELS_DIR/<name>. Optional: an
# absolute path passed to run_eval.sh always works regardless of this.
export MODELS_DIR=/path/to/your/models
