#!/usr/bin/env bash
# Matched-budget GRPO arm for ES-capacity. See docs/matched-experiment.md.
#
#   ./run_grpo_matched.sh --smoke    3 steps, ~20 min, acceptance gate before the long run
#   ./run_grpo_matched.sh            full 51-step run, ~5h
#   ./run_grpo_matched.sh --resume   resume from the latest checkpoint
#
# Every non-default value here is deliberate and justified in the doc. In particular the
# deviations from SimpleRL-Zoo's own recipe (KL off, entropy off, lr scheduler constant,
# seq-mean-token-mean, no stop_token_ids) exist because this run is a controlled comparison
# against an ES arm that has no analogue for any of those knobs.
#
# data.val_files: verl always constructs a validation dataset, even with
# val_before_train=False and test_freq=-1. Left unset it falls back to
# ~/data/rlhf/gsm8k/test.parquet and the run dies at startup, so it is pointed at
# the train parquet below -- it is never actually read.
set -euo pipefail

VENV=${VENV:-/venv/train}
REPO=${REPO:-/workspace/repos/ES-capacity}
DATA=${DATA:-/workspace/data/lvl3to5_es_template/train.parquet}
CKPT=${CKPT:-/workspace/ckpt/grpo-1.5b-matched}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B}
export ES_AT_SCALE_PATH=${ES_AT_SCALE_PATH:-/workspace/repos/es-at-scale}

# Memory knobs. Defaults target the 8x RTX 3060 12GB box this repo currently runs on;
# the doc's original values assumed 8x RTX 4090 48GB. None of these change the
# optimization math: with use_dynamic_bsz=True and ppo_mini_batch_size == train_batch_size,
# *_max_token_len_per_gpu only sets micro-batch splitting inside a single gradient update.
# TOKGPU must stay >= data.max_prompt_length + data.max_response_length (4096) or the run
# dies in compute_log_prob. Raise TOKGPU on larger cards for throughput.
GPUMEM=${GPUMEM:-0.45}
TOKGPU=${TOKGPU:-4096}
MAXBT=${MAXBT:-4096}
MAXSEQ=${MAXSEQ:-64}
OFFLOAD=${OFFLOAD:-True}

SMOKE=0; RESUME_MODE=disable
for a in "$@"; do
  case "$a" in
    --smoke)  SMOKE=1 ;;
    --resume) RESUME_MODE=auto ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

if [[ $SMOKE -eq 1 ]]; then
  STEPS=3; SAVE_FREQ=3; RUN_NAME=grpo-1.5b-matched-smoke; CKPT="${CKPT}-smoke"
else
  STEPS=50; SAVE_FREQ=10; RUN_NAME=grpo-1.5b-matched
fi

# verl starts its own Ray head; the image ships one that would collide on :6379.
# vllm/model-ui hold GPU memory that distorts vLLM's gpu_memory_utilization accounting
# (it is computed against TOTAL device memory, not free memory).
supervisorctl stop vllm model-ui ray >/dev/null 2>&1 || true

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

source "${VENV}/bin/activate"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.norm_adv_by_std_in_grpo=True \
  \
  data.train_files="${DATA}" \
  data.val_files="${DATA}" \
  data.train_batch_size=256 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=True \
  \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${TOKGPU} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.calculate_entropy=True \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=32 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPUMEM} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAXBT} \
  actor_rollout_ref.rollout.max_num_seqs=${MAXSEQ} \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${TOKGPU} \
  \
  custom_reward_function.path="${REPO}/scripts/es_reward_verl.py" \
  custom_reward_function.name=compute_score \
  \
  trainer.total_training_steps=${STEPS} \
  trainer.total_epochs=2 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.resume_mode=${RESUME_MODE} \
  trainer.default_local_dir="${CKPT}" \
  trainer.project_name=es-capacity \
  trainer.experiment_name=${RUN_NAME} \
  trainer.logger='["console","tensorboard"]' \
  actor_rollout_ref.nccl_timeout=1800
