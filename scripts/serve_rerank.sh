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
MODEL="${RERANK_MODEL:-BAAI/bge-m3}"
HOST="${RERANK_HOST:-127.0.0.1}"
PORT="${RERANK_PORT:-8002}"
API_KEY="${RERANK_API_KEY:-EMPTY}"
VLLM_BIN="${VLLM_BIN:-vllm}"
DTYPE="${RERANK_DTYPE:-auto}"
CUDA_DEVICES="${RERANK_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
DEFAULT_TENSOR_PARALLEL_SIZE="$(
  IFS=,
  read -ra devices <<< "${CUDA_DEVICES}"
  echo "${#devices[@]}"
)"
TENSOR_PARALLEL_SIZE="${RERANK_TENSOR_PARALLEL_SIZE:-${DEFAULT_TENSOR_PARALLEL_SIZE}}"
GPU_MEMORY_UTILIZATION="${RERANK_GPU_MEMORY_UTILIZATION:-0.10}"
MAX_MODEL_LEN="${RERANK_MAX_MODEL_LEN:-8192}"
TASK="${RERANK_TASK:-score}"
CHAT_TEMPLATE="${RERANK_CHAT_TEMPLATE:-}"
HF_OVERRIDES="${RERANK_HF_OVERRIDES:-}"
EXTRA_ARGS="${RERANK_VLLM_ARGS:-}"

if [[ -z "${CHAT_TEMPLATE}" && "${MODEL}" == *"Qwen3-Reranker"* ]]; then
  CHAT_TEMPLATE="${ROOT_DIR}/scripts/templates/qwen3_reranker.jinja"
fi

if [[ -z "${HF_OVERRIDES}" && "${MODEL}" == *"Qwen3-Reranker"* ]]; then
  HF_OVERRIDES='{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'
fi

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
  --max-model-len "${MAX_MODEL_LEN}"
  --runner pooling
  --task "${TASK}"
  --disable-frontend-multiprocessing
)

if [[ -n "${HF_OVERRIDES}" ]]; then
  args+=(--hf_overrides "${HF_OVERRIDES}")
fi

if [[ -n "${CHAT_TEMPLATE}" ]]; then
  args+=(--chat-template "${CHAT_TEMPLATE}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # Intentionally split extra vLLM args supplied by the caller.
  # shellcheck disable=SC2206
  extra=(${EXTRA_ARGS})
  args+=("${extra[@]}")
fi

echo "Starting rerank model ${MODEL} on http://${HOST}:${PORT}/v1 with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, tensor_parallel_size=${TENSOR_PARALLEL_SIZE}, task=${TASK}"
exec "${args[@]}"
