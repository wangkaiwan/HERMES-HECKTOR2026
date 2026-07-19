# HECKTOR 2026 submission container, mirroring the docker-template branch
# of BioMedIA-MBZUAI/HECKTOR2026.

FROM --platform=linux/amd64 pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS hecktor2026-task
# cu12.8 (not 12.6) is needed for sm_120 (RTX 5090 Blackwell) kernels — our local
# test GPUs. cu12.8 still bundles sm_75 so the submission T4 keeps working.
# Matches the training env (brainiac: torch 2.8.0+cu128).

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Minimal OS deps for SimpleITK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Grand Challenge runs the container as a non-root user with UID 1000.
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user docker/requirements-docker.txt /opt/app/requirements.txt
RUN python -m pip install --user --no-cache-dir --no-color -r /opt/app/requirements.txt

# Project source — needed by inference.py for model / data / transforms imports.
COPY --chown=user:user models   /opt/app/models
COPY --chown=user:user training /opt/app/training
COPY --chown=user:user data     /opt/app/data
COPY --chown=user:user configs  /opt/app/configs

# Inference entry point + medai_hybrid helper modules
COPY --chown=user:user docker/inference.py /opt/app/inference.py
COPY --chown=user:user docker/medai_patch.py /opt/app/medai_patch.py
COPY --chown=user:user docker/radio_extract.py /opt/app/radio_extract.py

# Version label — bumps the image content hash (GC dedupes images by SHA256) so a
# new submission is accepted, without changing any runtime behavior. v8 = 10-fold
# seg ensemble (30 ckpts); code identical to v7.2.
LABEL hecktor.version="v15-survloss-sigmoid5exp-2026-07-14"

ENTRYPOINT ["python", "inference.py"]
