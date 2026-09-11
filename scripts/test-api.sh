#!/usr/bin/env bash
set -euo pipefail
URL="${URL:-http://127.0.0.1:8002}"
curl -s "${URL}/v1/models"
echo
time curl -s "${URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen38","prompt":"Кратко объясни, что такое квантование нейросетей.","max_tokens":100,"temperature":0.2}'
echo
