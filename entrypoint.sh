#!/usr/bin/env bash
set -e

MODEL="${MODEL:-/data/models/Qwen3.8-27B-GPTQ-Int4}"
MODEL_NAME="${MODEL_NAME:-qwen38}"
PORT="${PORT:-8002}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8207}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-26000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"

extra_args=(
  --enable-prefix-caching
  --enable-chunked-prefill
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER:-hermes}"
  --chat-template-content-format string
)

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  extra_args+=(--enforce-eager)
fi

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${MODEL_NAME}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gdn-prefill-backend "${GDN_PREFILL_BACKEND}" \
  "${extra_args[@]}"
