#!/bin/bash
# Launch the GR00T inference policy server (ZMQ) that RoboCasa rollout
# clients connect to. Also usable standalone for any client speaking the
# same protocol.
#
# Usage:
#   scripts/run_groot_server.sh [model_path] [embodiment_tag] [port]
#
# Default model: nvidia/GR00T-N1.6-3B, NOT N1-2B (referenced elsewhere in this
# repo's configs/compsteer.yaml for the separate ManiSkill CompSteer eval) and
# NOT N1.7-3B (an earlier version of this script used it as a workaround for
# an unrelated loader bug, see below). Isaac-GR00T is pinned to n1.6.1-release
# in the Dockerfile specifically to pair with this checkpoint.
#
# Why N1.6-3B: EmbodimentTag.ROBOCASA_PANDA_OMRON is only in each checkpoint's
# actually-*trained* embodiment set (checkpoint's statistics.json — a global
# embodiment_id.json listing the tag exists in every release since n1.6, but
# that's just a shared ID vocabulary, not a claim the checkpoint was trained
# on it). Verified via HF:
#   nvidia/GR00T-N1.6-3B  statistics.json -> behavior_r1_pro, gr1, robocasa_panda_omron  (yes)
#   nvidia/GR00T-N1.7-3B  statistics.json -> xdof*, oxe_droid*, real_g1*, real_r1_pro_sharpa*  (no robocasa)
# N1.7-3B was originally chosen here only to dodge a *different* problem:
# Isaac-GR00T's unpinned main branch dropped the "gr00t_n1" model_type (used
# by N1-2B) in favor of "Gr00tN1d7" — confirmed KeyError: 'gr00t_n1' when
# loading N1-2B against unpinned main. Pinning to n1.6.1-release (model_type
# "Gr00tN1d6") avoids that problem too, without giving up RoboCasa support.
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/workspace/Isaac-GR00T}"
MODEL_PATH="${1:-nvidia/GR00T-N1.6-3B}"
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
