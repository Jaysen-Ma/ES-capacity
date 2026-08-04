#!/bin/bash

#SBATCH --job-name=4b_base-pop16k-n16
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1

# --- Create logs directory if it doesn't exist ---
LOG_DIR="./hyperscale-es-vllm/logs/"
mkdir -p "$LOG_DIR"

echo "---------------------------------"
echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "Nodes involved: $SLURM_JOB_NODELIST"
echo "Running on GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Number of GPUs per node: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Log file: $LOG_DIR/multinode_n16-$SLURM_JOB_ID.log"
echo "---------------------------------"

# -----------------------------------------
# User-settable parameters
# -----------------------------------------
sigma="0.001"
learning_rate="0.0002"
max_tokens="4096"
model_name="Qwen/Qwen3-4B"
population_size="16384"
steps_per_adapter="4"
lora_r="1"
task="math:deepscaler40k"
normalize_with_std="normalize-with-std"
# set normalize_with_std="normalize-with-std" to enable, and ="" to disable
scale_lr_in_grad=""
# set scale_lr_in_grad="scale-lr-in-grad" to enable, and ="" to disable
prompt_batch_size="16"
samples_per_prompt="1"
temperature="0.0"
pass_at_k="no-pass-at-k"
# set pass_at_k="pass-at-k" to enable, and "no-pass-at-k" to disable
steps_per_eval="1"
sub_dataset_size="null"
# Set to "null" or "None" or empty string to use full dataset
name_prefix="4b_base-pop16k-n16"

# -----------------------------------------

# --- Echo parameters for logging ---
echo "Parameters:"
echo "  sigma: $sigma"
echo "  learning_rate: $learning_rate"
echo "  max_tokens: $max_tokens"
echo "  model_name: $model_name"
echo "  population_size: $population_size"
echo "  steps_per_adapter: $steps_per_adapter"
echo "  lora_r: $lora_r"
echo "  task: $task"
echo "  normalize_with_std: $normalize_with_std"
echo "  scale_lr_in_grad: $scale_lr_in_grad"
echo "  prompt_batch_size: $prompt_batch_size"
echo "  samples_per_prompt: $samples_per_prompt"
echo "  temperature: $temperature"
echo "  pass_at_k: $pass_at_k"
echo "  steps_per_eval: $steps_per_eval"
echo "  sub_dataset_size: $sub_dataset_size"
echo "  name_prefix: $name_prefix"
echo "---------------------------------"

if [[ "$sub_dataset_size" == "None" ]] || [[ "$sub_dataset_size" == "null" ]] || [[ -z "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD=""
else
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

# --- Activate Environment ---
echo "Activating virtual environment..."
source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"

# --- Set WandB directory ---
export WANDB_DIR="$SCRATCH/for_esvllm/wandb"

# --- Allow expandable PyTorch memory --- 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Force Hugging Face to use offline mode (avoid rate limiting) ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Conditionally set compile caches based on model size ---
# Extracts the numeric value immediately before a standalone 'B' in the model name
# e.g. "Qwen3-4B" -> 4, "Meta-Llama-3-70B" -> 70, "mistral-7b" -> 7
get_model_size_b() {
    local name="$1"
    local size
    size=$(echo "$name" | grep -oiP '\d+(?=b\b)' | tail -1)
    echo "${size:-0}"
}

model_size_b=$(get_model_size_b "$model_name")

if [ "$model_size_b" -gt 0 ] && [ "$model_size_b" -lt 14 ]; then
    echo "Model size is ${model_size_b}B (< 14B): enabling job-specific compile caches"
    export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_${SLURM_JOB_ID}"
    export TRITON_CACHE_DIR="$SCRATCH/.triton_cache_${SLURM_JOB_ID}"
    export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_cache_${SLURM_JOB_ID}"
    mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

    cleanup_caches() {
        echo "Cleaning up job-specific caches..."
        rm -rf "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
        echo "Cache cleanup complete"
    }
    trap cleanup_caches EXIT
else
    echo "Model size is ${model_size_b}B (>= 14B or unparseable): skipping job-specific compile caches"
fi

# --- Change to Working Directory ---
echo "Changing to working directory..."
cd "$HOME/hyperscale/hyperscale-es-vllm" || exit 1

# --- Clean up leftover shared memory directories from previous jobs (on all nodes) ---
echo "Cleaning up /dev/shm from previous jobs on all nodes..."
echo "Current job ID: $SLURM_JOB_ID"
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" bash -c '
    echo "$(hostname): Cleaning /dev/shm..."
    chmod -R u+rwx /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    echo "$(hostname): Cleanup complete"
'
echo "Cleanup complete on all nodes"

# ==========================================
# === RAY CLUSTER SETUP (MULTI-NODE) ===
echo "Setting up Ray Cluster..."

# 1. Get the list of nodes and the head node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 2. Port configuration
port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head

echo "Head node: $head_node ($head_node_ip)"
echo "Ray Head IP: $ip_head"

# 3. Start Ray Head on the primary node
echo "Starting Ray Head on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${SLURM_GPUS_PER_NODE}" --block &

# 4. Wait briefly for head to initialize
sleep 10

# 5. Start Ray Workers on the remaining nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ $worker_num -gt 0 ]; then
    for ((i=1; i<=worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Ray Worker on $node_i..."
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${SLURM_GPUS_PER_NODE}" --block &
    done
fi

# 6. Wait for all nodes to register
echo "Waiting for Ray workers to connect..."
sleep 20
python -c "import ray; ray.init(address='auto'); print('Ray Cluster Resources:', ray.cluster_resources())"
# ==========================================


# --- Run the Python Script (Head Node Only) ---
echo "Starting Python script..."

# Build flag strings for optional flags (only add if non-empty)
NORMALIZE_FLAG=""
if [[ -n "$normalize_with_std" ]]; then
    NORMALIZE_FLAG="--${normalize_with_std}"
fi

SCALE_LR_FLAG=""
if [[ -n "$scale_lr_in_grad" ]]; then
    SCALE_LR_FLAG="--${scale_lr_in_grad}"
fi

PASSATK_FLAG=""
if [[ -n "$pass_at_k" ]]; then
    PASSATK_FLAG="--${pass_at_k}"
fi

python es_lora_multinode.py \
    --sigma "$sigma" \
    --learning-rate "$learning_rate" \
    --max-tokens "$max_tokens" \
    --model-name "$model_name" \
    --population-size "$population_size" \
    --steps-per-adapter "$steps_per_adapter" \
    --lora-r "$lora_r" \
    --task "$task" \
    $NORMALIZE_FLAG \
    $SCALE_LR_FLAG \
    --prompt-batch-size "$prompt_batch_size" \
    --samples-per-prompt "$samples_per_prompt" \
    --temperature "$temperature" \
    $PASSATK_FLAG \
    --steps-per-eval "$steps_per_eval" \
    $DATASET_SIZE_CMD \
    --name-prefix "$name_prefix" \
    --use-wandb

PYTHON_EXIT_CODE=$?
echo "---------------------------------"
if [ $PYTHON_EXIT_CODE -eq 124 ]; then
    echo "Job timed out after 3 hours"
elif [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Job finished with error code $PYTHON_EXIT_CODE"
else
    echo "Job finished successfully"
fi
echo "---------------------------------"

# Clean up Ray cluster
echo "Stopping Ray cluster..."
ray stop || true
echo "Ray cluster stopped"

# Clean up shared memory directories on all nodes (best effort)
echo "Cleaning up /dev/shm directories on all nodes..."
if [ -n "$SLURM_JOB_NUM_NODES" ] && [ -n "$SLURM_JOB_NODELIST" ]; then
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" bash -c '
        chmod -R u+rwx /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
        rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    ' || true
else
    chmod -R u+rwx /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
fi
echo "Shared memory cleanup complete"