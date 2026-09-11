# Qwen3.8-27B GPTQ Int4 on NVIDIA Tesla V100

Docker runtime for `Max73333/Qwen3.8-27B-GPTQ-Int4-V100` on NVIDIA Tesla V100 (`sm_70`) using a patched source build of vLLM.

## Requirements

- Linux
- Docker
- NVIDIA driver
- NVIDIA Container Toolkit
- NVIDIA Tesla V100 or compatible NVIDIA GPU

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## Quick start

```bash
git clone https://github.com/Max73333/qwen38-v100.git
cd qwen38-v100
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f
```

API:

```bash
curl http://127.0.0.1:8002/v1/models
```

```bash
curl http://127.0.0.1:8002/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen38",
    "prompt": "Кратко объясни, что такое квантование нейросетей.",
    "max_tokens": 100,
    "temperature": 0.2
  }'
```

## Tested configuration

```text
GPU: Tesla V100-SXM2-32GB
Compute capability: sm_70
CUDA: 12.8
PyTorch: 2.10.0+cu128
vLLM source commit: bcf2be9
GDN prefill backend: Triton
Quantization: GPTQ Int4, group_size=128, desc_act=false, sym=true
```

Observed after warmup:

- 100 generated tokens: ~2.7 s
- 500 generated tokens: ~12.5 s
- steady-state generation: ~37-40 tok/s

The first request is slower because Triton kernels and CUDA graphs warm up.

## Important: regenerate the exact patches before publishing

The patch files included here are reconstructed from the known working changes.

On the actual working host, generate the authoritative diffs:

```bash
chmod +x scripts/export-local-patches.sh
./scripts/export-local-patches.sh ~/vllm-018 "$(pwd)/patches"
```

Then review:

```bash
less patches/qwen38-text-only.patch
less patches/kv-cache-v100.patch
```

Make sure no temporary debug changes are included.

## Configuration

`.env`:

```env
MODEL=Max73333/Qwen3.8-27B-GPTQ-Int4-V100
MODEL_NAME=qwen38
PORT=8002
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=1
GDN_PREFILL_BACKEND=triton
```

## Why the patches are needed

`qwen38-text-only.patch` contains the compatibility routing used for the Qwen3.8 text-only model in this vLLM checkout.

`kv-cache-v100.patch` works around the ambiguous `[2, 2, ...]` hybrid KV-cache layout encountered during CUDA Graph profiling. Without the workaround, the tested build failed while deciding whether the layout was `(2, num_blocks, ...)` or `(num_blocks, 2, ...)`.

## Publish to GHCR

The repository contains `.github/workflows/publish-ghcr.yml`.

After pushing the repository to GitHub:

```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions will build:

```text
ghcr.io/OWNER/qwen38-v100:latest
ghcr.io/OWNER/qwen38-v100:v0.1.0
```

Then users can run the prebuilt image:

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  -p 8002:8002 \
  -v qwen38-hf-cache:/root/.cache/huggingface \
  ghcr.io/OWNER/qwen38-v100:latest
```

Model weights stay on Hugging Face and are downloaded on first run.
