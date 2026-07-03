#!/bin/bash
# Launch the GR00T inference policy server (ZMQ) that RoboCasa rollout
# clients connect to. Also usable standalone for any client speaking the
# same protocol.
#
# Usage:
#   scripts/run_groot_server.sh [model_path] [embodiment_tag] [port]
#
# Defaults match configs/compsteer.yaml's active_model (groot / N1-2B).
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/workspace/Isaac-GR00T}"
MODEL_PATH="${1:-nvidia/GR00T-N1-2B}"
EMBODIMENT_TAG="${2:-ROBOCASA_PANDA_OMRON}"
PORT="${3:-5555}"

echo "======================================"
echo "GR00T policy server"
echo "  Model:      $MODEL_PATH"
echo "  Embodiment: $EMBODIMENT_TAG"
echo "  Port:       $PORT"
echo "======================================"

cd "$GR00T_ROOT"
# --no-sync: the venv was already fully built during the image build (uv
# sync). Skip uv's automatic re-sync/rebuild check here — it tries to touch
# gr00t.egg-info timestamps, which fails under Apptainer's default read-only
# squashfs image filesystem.
exec uv run --no-sync python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --use-sim-policy-wrapper \
    --port "$PORT"
