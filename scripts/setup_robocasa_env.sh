#!/bin/bash
# One-time (idempotent) setup of the isolated RoboCasa venv + kitchen assets.
#
# RoboCasa's `robosuite` dependency conflicts with GR00T's pinned deps, so
# Isaac-GR00T's own setup script creates a dedicated venv for it. Kitchen
# assets are ~10GB, so this is a runtime step against a persistent volume
# rather than something baked into the Docker image.
#
# Usage:
#   scripts/setup_robocasa_env.sh
#
# Re-running is safe: it skips work once the completion marker exists.
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/workspace/Isaac-GR00T}"
ROBOCASA_VENV="${GR00T_ROOT}/gr00t/eval/sim/robocasa/robocasa_uv/.venv"
MARKER="${ROBOCASA_VENV}/.setup_complete"

if [ -f "$MARKER" ]; then
    echo "RoboCasa environment already set up at $ROBOCASA_VENV — skipping."
    exit 0
fi

cd "$GR00T_ROOT"

echo "Setting up RoboCasa venv (robosuite + robocasa)..."
# setup_RoboCasa.sh (Isaac-GR00T's own script) already installs everything and
# downloads the kitchen assets itself (echo y | python download_kitchen_assets.py,
# with macro setup as a side effect of that) — do NOT re-run either step here.
# A second, unpiped call to download_kitchen_assets crashes with EOFError when
# run non-interactively (confirmed: input() has no stdin under apptainer exec),
# and since this script uses `set -e`, that crash aborted before touch "$MARKER"
# ever ran, even though the venv itself had already been built successfully.
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh

touch "$MARKER"
echo "RoboCasa environment ready → $ROBOCASA_VENV"
