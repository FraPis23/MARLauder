# PyTorch + CUDA runtime base (torch & CUDA libs preinstalled).
# CUDA 12.8 + torch cu128 covers both RTX 4080 (Ada, sm_89) and RTX 5080 (Blackwell, sm_120).
# Blackwell requires host driver >= 570.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal tooling: git for pip dependencies, libgl for headless matplotlib/imageio
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MARLauder

# torch is already in the base image; pip sees the requirement as satisfied.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Keep the container alive so several shells can attach (docker compose exec).
CMD ["sleep", "infinity"]
