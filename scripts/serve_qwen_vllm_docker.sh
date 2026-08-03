#!/usr/bin/env bash
# Launch a Dockerized vLLM OpenAI server for ES-capacity eval (GB10 / cu130).
#
# Paths and vLLM knobs come from config.toml (see config.sample.toml).
#
# Usage:
#   bash scripts/serve_qwen_vllm_docker.sh              # models.base from config
#   bash scripts/serve_qwen_vllm_docker.sh instruct     # models.instruct
#   bash scripts/serve_qwen_vllm_docker.sh base 8001    # optional port override
#   bash scripts/serve_qwen_vllm_docker.sh /abs/path/to/model [served-name] [port]
#
# Then evaluate:
#   python scripts/eval_aime_passk.py --model-key base
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

cfg() {
  python -m es_capacity.config "$1"
}

IMAGE="$(cfg vllm.image)"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-$(cfg vllm.gpu_memory_utilization)}"
MAX_LEN="${MAX_MODEL_LEN:-$(cfg vllm.max_model_len)}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$(cfg vllm.max_num_seqs)}"
CONTAINER_NAME="${4:-$(cfg vllm.container_name)}"
DEFAULT_PORT="$(cfg vllm.port)"

ARG1="${1:-base}"
ARG2="${2:-}"
ARG3="${3:-}"

if [[ "${ARG1}" == /* || "${ARG1}" == ~* ]]; then
  MODEL_HOST_PATH="${ARG1}"
  SERVED_NAME="${ARG2:-$(basename "${MODEL_HOST_PATH}" | tr '[:upper:]' '[:lower:]')}"
  PORT="${ARG3:-${DEFAULT_PORT}}"
elif [[ "${ARG1}" == "base" || "${ARG1}" == "instruct" ]]; then
  MODEL_HOST_PATH="$(cfg "model.${ARG1}.path")"
  SERVED_NAME="$(cfg "model.${ARG1}.served_name")"
  # optional: serve_qwen_vllm_docker.sh instruct 8001
  if [[ -n "${ARG2}" && "${ARG2}" =~ ^[0-9]+$ ]]; then
    PORT="${ARG2}"
  else
    PORT="${ARG3:-${DEFAULT_PORT}}"
  fi
else
  echo "Usage: $0 [base|instruct|/abs/model/path] [served-name|port] [port] [container-name]" >&2
  echo "Configure models in config.toml (copy from config.sample.toml)." >&2
  exit 1
fi

if [[ ! -d "${MODEL_HOST_PATH}" ]]; then
  echo "Model directory not found: ${MODEL_HOST_PATH}" >&2
  exit 1
fi

MODEL_BASENAME="$(basename "${MODEL_HOST_PATH}")"
MODELS_PARENT="$(cd "$(dirname "${MODEL_HOST_PATH}")" && pwd)"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing existing container ${CONTAINER_NAME}…"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting ${CONTAINER_NAME}"
echo "  config:  $(cfg config_path)"
echo "  image:   ${IMAGE}"
echo "  model:   ${MODEL_HOST_PATH} -> /models/${MODEL_BASENAME}"
echo "  served:  ${SERVED_NAME}"
echo "  port:    ${PORT} (host network)"
echo "  max_len: ${MAX_LEN}  gpu_mem: ${GPU_MEM}  max_num_seqs: ${MAX_NUM_SEQS}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --network host \
  -v "${MODELS_PARENT}:/models" \
  "${IMAGE}" \
  "/models/${MODEL_BASENAME}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --trust-remote-code \
  --enable-prefix-caching

echo
echo "Wait until ready, then:"
echo "  curl -s http://127.0.0.1:${PORT}/v1/models | jq ."
echo "Logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
