#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

export HF_HOME="${HF_HOME:-/data1/wujinqi/huggingface}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
MODEL="${ANSWER_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${ANSWER_HOST:-127.0.0.1}"
PORT="${ANSWER_PORT:-8000}"
API_KEY="${ANSWER_API_KEY:-${LOCAL_LLM_API_KEY:-EMPTY}}"
VLLM_BIN="${VLLM_BIN:-vllm}"
DTYPE="${ANSWER_DTYPE:-auto}"
CUDA_DEVICES="${ANSWER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
DEFAULT_TENSOR_PARALLEL_SIZE="$(
  IFS=,
  read -ra devices <<< "${CUDA_DEVICES}"
  echo "${#devices[@]}"
)"
TENSOR_PARALLEL_SIZE="${ANSWER_TENSOR_PARALLEL_SIZE:-${DEFAULT_TENSOR_PARALLEL_SIZE}}"
GPU_MEMORY_UTILIZATION="${ANSWER_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${ANSWER_MAX_MODEL_LEN:-262144}"
EXTRA_ARGS="${ANSWER_VLLM_ARGS:-}"

if [[ "${VLLM_BIN}" == "vllm" && -x "${ROOT_DIR}/.venv/bin/vllm" ]]; then
  VLLM_BIN="${ROOT_DIR}/.venv/bin/vllm"
elif [[ "${VLLM_BIN}" == "vllm" && -x "/home/wujinqi/miniconda3/envs/deepresearch/bin/vllm" ]]; then
  VLLM_BIN="/home/wujinqi/miniconda3/envs/deepresearch/bin/vllm"
fi

args=(
  "${VLLM_BIN}" serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --disable-frontend-multiprocessing
)

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  args+=(--max-model-len "${MAX_MODEL_LEN}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # Intentionally split extra vLLM args supplied by the caller.
  # shellcheck disable=SC2206
  extra=(${EXTRA_ARGS})
  args+=("${extra[@]}")
fi

echo "Starting answer model on http://${HOST}:${PORT}/v1 with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} and tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
exec "${args[@]}"
