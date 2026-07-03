# syntax=docker/dockerfile:1
#
# CompSteer image: Isaac-GR00T (uv-managed) + ManiSkill + this repo, ready to
# evaluate a raw GR00T checkpoint on ManiSkill in-process, or to drive the
# official GR00T<->RoboCasa client/server harness.
#
# RoboCasa's own dependencies (robosuite) conflict with GR00T's pinned deps,
# so — following Isaac-GR00T's own setup — the RoboCasa venv and its ~10GB of
# kitchen assets are NOT baked into this image. They're created at container
# runtime into a persistent volume via scripts/setup_robocasa_env.sh (see
# docker-compose.yml's `robocasa-client` service).
#
# Build:
#   docker build -t compsteer-groot .
# Or via compose:
#   docker compose build

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    GR00T_ROOT=/workspace/Isaac-GR00T \
    GROOT_VENV=/workspace/Isaac-GR00T/.venv

# ── System dependencies ──────────────────────────────────────────────────
# ffmpeg/libsm6/libxext6: GR00T video decoding (torchcodec)
# libegl1-mesa-dev/libglu1-mesa/libgl1/libosmesa6-dev: headless rendering for
#   RoboCasa (MuJoCo/robosuite) and general OpenGL fallback
# libvulkan1/vulkan-tools: ManiSkill/SAPIEN rendering (Vulkan ICD is mounted
#   in at runtime by the NVIDIA container toolkit when --gpus all is used)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git git-lfs curl ca-certificates \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0 \
        libegl1-mesa-dev libglu1-mesa libgl1 libosmesa6-dev patchelf \
        libvulkan1 vulkan-tools \
        python3.10 python3.10-venv python3.10-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── uv (Isaac-GR00T's package manager) ───────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

# ── Isaac-GR00T ───────────────────────────────────────────────────────────
RUN git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git "${GR00T_ROOT}"
WORKDIR ${GR00T_ROOT}
RUN uv sync --python 3.10

# uv-managed venvs don't ship pip, so use `uv pip` (targeting the GR00T venv
# explicitly) rather than "<venv>/bin/python -m pip".
ENV VIRTUAL_ENV=${GROOT_VENV} \
    PATH="${GROOT_VENV}/bin:${PATH}"

# ── This repo (compsteer), including ManiSkill via requirements.txt ─────
# (no known dependency conflict between ManiSkill and GR00T's own deps)
WORKDIR /workspace/compsteer
COPY requirements.txt setup.py ./
RUN uv pip install --python "${GROOT_VENV}/bin/python" --no-cache-dir -r requirements.txt
COPY . .
RUN uv pip install --python "${GROOT_VENV}/bin/python" --no-cache-dir -e .

WORKDIR /workspace/compsteer

CMD ["/bin/bash"]
