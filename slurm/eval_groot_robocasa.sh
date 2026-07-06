#!/bin/bash
#SBATCH --job-name=groot_robocasa_eval
#SBATCH --output=logs/robocasa_eval_%j.out
#SBATCH --error=logs/robocasa_eval_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuv
# ^ set to a real partition name for this cluster, e.g.:
#     sinfo -o "%P %G %l %D %N"
#
# Raw GR00T (no CompSteer steering) evaluation on RoboCasa, run via the
# compsteer-groot.sif Apptainer image on HPC.
#
# Mirrors the groot-server-robocasa / robocasa-client split in
# docker-compose.yml, but as ONE SLURM job on one GPU node (SLURM allocates
# per-node here, not per-service): the GR00T policy server and the RoboCasa
# client both run from the same .sif, talking over localhost ZMQ.
#
# Prerequisites:
#   - compsteer-groot.sif already present at /work/hezhang/docker_images/
#     (built locally via `apptainer build ... docker-archive://...`, see
#     project history / README — HPC's BeeGFS /work mount can't do the
#     image build itself, so it's built elsewhere and copied in as a .sif)
#   - This repo checked out at /work/hezhang/latentSteerVLA (edit REPO below
#     if different)
#   - Submit from the repo root so relative --output/--error paths resolve
#     correctly:
#       cd /work/hezhang/latentSteerVLA
#       sbatch slurm/eval_groot_robocasa.sh [n_episodes] [task ...]
#
# Examples:
#   sbatch slurm/eval_groot_robocasa.sh                          # 20 episodes, all tasks
#   sbatch slurm/eval_groot_robocasa.sh 20 all
#   sbatch slurm/eval_groot_robocasa.sh 10 open_drawer coffee_press_button
#
# Known unverified assumptions (check the job log for these on first run):
#   - Compute nodes on this cluster may not have internet access, in which
#     case the GR00T-N1.6-3B checkpoint download from Hugging Face will hang
#     or fail. If so, pre-download it on the login node into $HF_CACHE
#     first (see bottom of this file) and re-run with the cache warm.
#   - The RoboCasa venv setup (scripts/setup_robocasa_env.sh) does its own
#     package installs on top of $ROBOCASA_DATA, which lives on BeeGFS.
#     Apptainer's own unprivileged *image builds* fail on this filesystem
#     (hardlink semantics) — plain `uv`/`pip` installs are a different code
#     path and are expected to be fine, but haven't been verified here.
#     UV_LINK_MODE=copy is set defensively below in case uv's own hardlink
#     optimization hits the same class of issue.

set -uo pipefail   # not -e: cleanup() must still run on failure paths

# ── Paths — edit if your layout differs ────────────────────────────────────
SIF="/work/hezhang/docker_images/compsteer-groot.sif"
REPO="/work/hezhang/latentSteerVLA"
ROBOCASA_DATA="/work/hezhang/robocasa_data"   # persistent isolated venv + ~10GB kitchen assets
HF_CACHE="/work/hezhang/hf_cache"             # persistent GR00T checkpoint cache (NOT $HOME — quota)
SCRATCH="/tmp/${SLURM_JOB_ID:-manual}"        # node-local scratch for apptainer's own tmp/cache

MODEL_PATH="nvidia/GR00T-N1.6-3B"   # NOT N1.7-3B (its statistics.json doesn't
                                     # include robocasa_panda_omron, despite the
                                     # tag existing in its EmbodimentTag enum) and
                                     # NOT N1-2B (Isaac-GR00T is pinned to
                                     # n1.6.1-release in the Dockerfile to match
                                     # this checkpoint's Gr00tN1d6 architecture —
                                     # see scripts/run_groot_server.sh)
EMBODIMENT_TAG="ROBOCASA_PANDA_OMRON"
PORT=5555

N_EPISODES="${1:-20}"
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -eq 0 ]; then
    TASKS=(all)
else
    TASKS=("$@")
fi

mkdir -p logs "$ROBOCASA_DATA" "$HF_CACHE" "$SCRATCH"

module load apptainer-1.4.1

export APPTAINER_TMPDIR="$SCRATCH/apptainer_tmp"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# Bind the live repo checkout over the image's baked-in copy (so config/
# script edits don't require rebuilding the .sif), the persistent RoboCasa
# venv+assets cache, and the persistent HF checkpoint cache.
#
# ROBOCASA_DATA is bound at /opt/robocasa_persist, NOT at robocasa_uv itself:
# setup_RoboCasa.sh (Isaac-GR00T's own script) unconditionally does
# `rm -rf robocasa_uv` on every invocation, and rm can't remove an active
# mount point ("Device or resource busy" — confirmed on a real run). See
# scripts/setup_robocasa_env.sh for how it builds into the ephemeral path on
# a cache miss, then copies into this persistent dir; and symlinks back out
# of it on a cache hit.
BINDS=(-B "${REPO}:/workspace/compsteer")
BINDS+=(-B "${ROBOCASA_DATA}:/opt/robocasa_persist")
BINDS+=(-B "${HF_CACHE}:/opt/hf_cache")

# NOTE: unlike N1.7-3B (which needed a token for the gated nvidia/Cosmos-Reason2-2B
# backbone), N1.6-3B does NOT require HF_TOKEN — confirmed by a live server run:
# it only fetches its own 2 safetensors shards ("Fetching 2 files"), and the
# "nvidia/Eagle-Block2A-2B-v2" name in its config.json is just descriptive
# metadata about the backbone architecture, not a separate download. HF_TOKEN
# is passed through below only in case a private/gated mirror is ever used.
APP_ENV=(--env "HF_HOME=/opt/hf_cache" --env "UV_LINK_MODE=copy" --env "ROBOCASA_PERSIST_DIR=/opt/robocasa_persist")
if [ -n "${HF_TOKEN:-}" ]; then
    APP_ENV+=(--env "HF_TOKEN=${HF_TOKEN}")
fi

# All `apptainer exec` calls below use --cleanenv: without it, apptainer passes
# the host shell's environment into the container by default, including any
# HPC environment-module vars like CUDA_HOME pointing at a host path (e.g.
# /opt/share/libs/intel/nvidia/cuda-12.8.0/) that doesn't exist inside the
# image. transformers opportunistically imports deepspeed at import time,
# which shells out to "$CUDA_HOME/bin/nvcc" to check compatibility — with a
# leaked host CUDA_HOME this is a FileNotFoundError that crashes the server
# before it even gets to load the model (confirmed on a real run). The image
# already has its own complete CUDA 12.8 toolchain baked in, so it doesn't
# need anything from the host environment. --cleanenv only blocks automatic
# passthrough of the calling shell's environment; the explicit --env flags in
# APP_ENV above still apply underneath it.

echo "======================================"
echo "GR00T x RoboCasa raw eval"
echo "  Node:       $(hostname)"
echo "  SIF:        $SIF"
echo "  Model:      $MODEL_PATH"
echo "  Embodiment: $EMBODIMENT_TAG"
echo "  Episodes:   $N_EPISODES"
echo "  Tasks:      ${TASKS[*]}"
echo "======================================"

# ── 1. Start the GR00T policy server in the background ────────────────────
SERVER_LOG="logs/robocasa_eval_${SLURM_JOB_ID:-manual}_server.log"
apptainer exec --nv --cleanenv --writable-tmpfs "${BINDS[@]}" "${APP_ENV[@]}" "$SIF" \
    bash /workspace/compsteer/scripts/run_groot_server.sh "$MODEL_PATH" "$EMBODIMENT_TAG" "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "Stopping GR00T server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# ── 2. Wait for the server socket to come up ───────────────────────────────
echo "Waiting for GR00T server on port $PORT..."
UP=0
for i in $(seq 1 60); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        echo "Server socket is up after $((i * 10))s."
        UP=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process died during startup. Log tail:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    sleep 10
done

if [ "$UP" -eq 0 ]; then
    echo "ERROR: server did not open port $PORT within 600s. Log tail:"
    tail -50 "$SERVER_LOG"
    exit 1
fi

# Socket accepting connections doesn't guarantee the model has finished
# loading yet — a short grace period before hammering it with rollouts.
sleep 30

# ── 3. One-time (idempotent) RoboCasa venv + kitchen asset setup ──────────
apptainer exec --nv --cleanenv --writable-tmpfs "${BINDS[@]}" "${APP_ENV[@]}" "$SIF" \
    bash /workspace/compsteer/scripts/setup_robocasa_env.sh

# ── 4. Run the RoboCasa client eval across the requested tasks ────────────
RESULTS_DIR="results/raw_groot/robocasa_${SLURM_JOB_ID:-manual}"
apptainer exec --nv --cleanenv --writable-tmpfs "${BINDS[@]}" "${APP_ENV[@]}" "$SIF" \
    python /workspace/compsteer/scripts/eval_groot_robocasa.py \
        --tasks "${TASKS[@]}" \
        --n_episodes "$N_EPISODES" \
        --policy_host 127.0.0.1 \
        --policy_port "$PORT" \
        --results_root "/workspace/compsteer/$RESULTS_DIR"

echo "Done. Results -> $REPO/$RESULTS_DIR"

# ── Optional: pre-warm the HF checkpoint cache from the login node ────────
# If compute nodes lack internet access, run this once on fe01 BEFORE
# submitting, so the job above finds the checkpoint already cached. No
# HF_TOKEN needed — nvidia/GR00T-N1.6-3B is fully public (confirmed via a live
# server run: it only fetches its own 2 safetensors shards, no separate
# gated backbone download):
#   module load apptainer-1.4.1
#   export HF_HOME=/work/hezhang/hf_cache
#   apptainer exec --nv -B /work/hezhang/hf_cache:/opt/hf_cache \
#       --env HF_HOME=/opt/hf_cache \
#       /work/hezhang/docker_images/compsteer-groot.sif \
#       python -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download('nvidia/GR00T-N1.6-3B')
#   "
