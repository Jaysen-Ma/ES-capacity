#!/bin/bash
# launch_job.sh

# -----------------------------------------
# User-settable parameters (edit these)
# -----------------------------------------
sigma="0.001"
learning_rate="0.001"
max_tokens="1024"
model_name="Qwen/Qwen3-4B"
population_size="256"
steps_per_adapter="4"
lora_r="1"
task="math:orz57k"
normalize_with_std="normalize-with-std"
scale_lr_in_grad="scale-lr-in-grad"
prompt_batch_size="16"
samples_per_prompt="1"
temperature="0.0"
pass_at_k=""
steps_per_eval="1"
sub_dataset_size="null"
name_prefix="docker-eggroll"

# set GPU index for running
GPU_DEVICES="6" 
# -----------------------------------------

# 1. Generate necessary folders
mkdir -p checkpoints logs wandb cache/huggingface

# 2. Build the image (Uses cache instantly if already built)
echo "Ensuring Docker image is built..."
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t ${USER}_eggroll_passatk_rebuttal .

# 3. Setup Optional Flags
DATASET_SIZE_CMD=""
if [[ "$sub_dataset_size" != "None" ]] && [[ "$sub_dataset_size" != "null" ]] && [[ -n "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

NORMALIZE_FLAG=$([[ -n "$normalize_with_std" ]] && echo "--${normalize_with_std}" || echo "")
SCALE_LR_FLAG=$([[ -n "$scale_lr_in_grad" ]] && echo "--${scale_lr_in_grad}" || echo "")
PASSATK_FLAG=$([[ -n "$pass_at_k" ]] && echo "--${pass_at_k}" || echo "")

# 4. Create the execution script for inside the container
cat << 'EOF' > run_inside_docker.sh
#!/bin/bash
# Route WandB to the mounted folder so local logs are also saved
export WANDB_DIR="/app/wandb"

echo "Cleaning up local shared memory..."
rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true

echo "Starting local Ray cluster..."
ray start --head --port=6379 --dashboard-host=0.0.0.0

echo "Starting Python training script..."
python es_lora_multinode.py \
    --sigma "$SIGMA" \
    --learning-rate "$LEARNING_RATE" \
    --max-tokens "$MAX_TOKENS" \
    --model-name "$MODEL_NAME" \
    --population-size "$POPULATION_SIZE" \
    --steps-per-adapter "$STEPS_PER_ADAPTER" \
    --lora-r "$LORA_R" \
    --task "$TASK" \
    $NORMALIZE_FLAG \
    $SCALE_LR_FLAG \
    --prompt-batch-size "$PROMPT_BATCH_SIZE" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --temperature "$TEMPERATURE" \
    $PASSATK_FLAG \
    --steps-per-eval "$STEPS_PER_EVAL" \
    $DATASET_SIZE_CMD \
    --name-prefix "$NAME_PREFIX" \
    --checkpoint-dir "/app/checkpoints" \
    --use-wandb

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE"

echo "Stopping Ray..."
ray stop || true
EOF
chmod +x run_inside_docker.sh

# 5. Launch the Docker Container
CONTAINER_NAME="${USER}_${name_prefix}_$(date +%s)"

echo "---------------------------------"
echo "Launching Job: $CONTAINER_NAME"
echo "Target GPUs: $GPU_DEVICES"
echo "---------------------------------"

# Run container. -v $(pwd):/app maps your host directory to the container.
# --shm-size=64g is required for vLLM & Ray shared memory.
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus "\"device=$GPU_DEVICES\"" \
    --shm-size=64g \
    -v $(pwd):/app \
    -e SIGMA="$sigma" \
    -e LEARNING_RATE="$learning_rate" \
    -e MAX_TOKENS="$max_tokens" \
    -e MODEL_NAME="$model_name" \
    -e POPULATION_SIZE="$population_size" \
    -e STEPS_PER_ADAPTER="$steps_per_adapter" \
    -e LORA_R="$lora_r" \
    -e TASK="$task" \
    -e PROMPT_BATCH_SIZE="$prompt_batch_size" \
    -e SAMPLES_PER_PROMPT="$samples_per_prompt" \
    -e TEMPERATURE="$temperature" \
    -e STEPS_PER_EVAL="$steps_per_eval" \
    -e NAME_PREFIX="$name_prefix" \
    -e NORMALIZE_FLAG="$NORMALIZE_FLAG" \
    -e SCALE_LR_FLAG="$SCALE_LR_FLAG" \
    -e PASSATK_FLAG="$PASSATK_FLAG" \
    -e DATASET_SIZE_CMD="$DATASET_SIZE_CMD" \
    -e WANDB_API_KEY="" \
    -e HF_TOKEN="" \
    ${USER}_eggroll /app/run_inside_docker.sh > "logs/${CONTAINER_NAME}.log" 2>&1

echo "Job launched in background."
echo "View stdout logs: tail -f logs/${CONTAINER_NAME}.log"
echo "View docker logs: docker logs -f $CONTAINER_NAME"