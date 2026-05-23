# PyTorch + CUDA runtime base (torch & CUDA libs preinstalled).
# CUDA 12.8 + torch cu128 => supporta sia RTX 4080 (Ada, sm_89)
# sia RTX 5080 (Blackwell, sm_120). Driver host >=570 richiesto per Blackwell.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Tool minimi (git per dipendenze, libgl per matplotlib/imageio headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/MARLauder

# torch gia presente nel base image; pip lo vede soddisfatto.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Container resta vivo per attach multipli (docker compose exec).
CMD ["sleep", "infinity"]
