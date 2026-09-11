FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_COMMIT=bcf2be9

# Только build args. Proxy НЕ сохраняем через ENV в итоговом образе.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=7.0
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}

# Используем локальную копию flash-attention, чтобы CMake не клонировал
# репозиторий и submodules во время сборки.
ENV VLLM_FLASH_ATTN_SRC_DIR=/opt/vendor/flash-attention

# ---------------------------------------------------------------------------
# System packages + Python 3.11
# ---------------------------------------------------------------------------

RUN export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}" no_proxy="${NO_PROXY}" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        git \
        curl \
        build-essential \
        pkg-config \
        libnuma-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------------------------

RUN python3.11 -m venv /opt/venv

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade \
        pip \
        setuptools \
        wheel \
        setuptools_scm \
        cmake \
        ninja

# ---------------------------------------------------------------------------
# Проверенная связка PyTorch / CUDA для Tesla V100
# ---------------------------------------------------------------------------

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# Критичные версии из реально работающего host venv
# ---------------------------------------------------------------------------

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        numpy==2.2.6 \
        transformers==4.57.6 \
        cuda-python==12.9.4 \
        cuda-bindings==12.9.4

# ---------------------------------------------------------------------------
# Остальные runtime dependencies из рабочего окружения
# requirements-runtime.txt НЕ должен содержать:
# vllm, torch, torchvision, torchaudio, numpy, transformers,
# cuda-python, cuda-bindings
# ---------------------------------------------------------------------------

COPY requirements-runtime.txt /tmp/requirements-runtime.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-runtime.txt

RUN pip uninstall -y nvidia-cudnn-cu13 || true

# После установки всех runtime deps проверяем, что resolver ничего не сломал.
RUN pip check

# ---------------------------------------------------------------------------
# vLLM source
# ---------------------------------------------------------------------------

RUN export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}" no_proxy="${NO_PROXY}" && \
    git clone https://github.com/vllm-project/vllm.git /opt/vllm && \
    cd /opt/vllm && \
    git checkout "${VLLM_COMMIT}"

WORKDIR /opt/vllm

# ---------------------------------------------------------------------------
# Наши рабочие патчи Qwen3.8 / V100
# ---------------------------------------------------------------------------

COPY patches/qwen38-text-only.patch /tmp/qwen38-text-only.patch
COPY patches/kv-cache-v100.patch /tmp/kv-cache-v100.patch

RUN git apply --check /tmp/qwen38-text-only.patch && \
    git apply /tmp/qwen38-text-only.patch && \
    git apply --check /tmp/kv-cache-v100.patch && \
    git apply /tmp/kv-cache-v100.patch

# ---------------------------------------------------------------------------
# Локальный flash-attention с уже загруженными submodules
# ---------------------------------------------------------------------------

COPY vendor/flash-attention /opt/vendor/flash-attention

# Проверяем, что критичные submodules действительно попали в build context.
RUN test -d /opt/vendor/flash-attention/csrc/cutlass && \
    test -d /opt/vendor/flash-attention/csrc/composable_kernel

# ---------------------------------------------------------------------------
# Build vLLM against existing torch
# ---------------------------------------------------------------------------

RUN python use_existing_torch.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . --no-build-isolation --no-deps

# ---------------------------------------------------------------------------
# Финальная проверка версий.
# Сборка должна упасть, если какая-либо зависимость опять изменила окружение.
# ---------------------------------------------------------------------------

RUN python - <<'PY'
import torch
import numpy
import transformers
import importlib.metadata as md

print("Python environment OK")
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
print("cuda-python:", md.version("cuda-python"))
print("cuda-bindings:", md.version("cuda-bindings"))

assert torch.__version__ == "2.10.0+cu128"
assert torch.version.cuda == "12.8"
assert numpy.__version__ == "2.2.6"
assert transformers.__version__ == "4.57.6"
assert md.version("cuda-python") == "12.9.4"
assert md.version("cuda-bindings") == "12.9.4"

print("Critical package versions OK")
PY

RUN pip check

# Не оставляем build-time временные файлы.
RUN rm -f \
    /tmp/qwen38-text-only.patch \
    /tmp/kv-cache-v100.patch \
    /tmp/requirements-runtime.txt

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8002

ENTRYPOINT ["/entrypoint.sh"]
