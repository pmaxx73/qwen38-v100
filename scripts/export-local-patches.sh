#!/usr/bin/env bash
set -euo pipefail
VLLM_DIR="${1:-$HOME/vllm-018}"
OUT_DIR="${2:-$(pwd)/patches}"
mkdir -p "${OUT_DIR}"
cd "${VLLM_DIR}"

git diff -- \
  vllm/model_executor/models/qwen3_5.py \
  vllm/config/model.py \
  vllm/renderers/base.py \
  vllm/model_executor/models/registry.py \
  vllm/model_executor/models/config.py \
  > "${OUT_DIR}/qwen38-text-only.patch"

git diff -- vllm/v1/worker/gpu_model_runner.py \
  > "${OUT_DIR}/kv-cache-v100.patch"

ls -lh "${OUT_DIR}"/*.patch
