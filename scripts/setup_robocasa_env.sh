#!/bin/bash
# One-time setup of the isolated RoboCasa venv + kitchen assets, cached in a
# persistent directory across separate job runs / container invocations.
#
# RoboCasa's `robosuite` dependency conflicts with GR00T's pinned deps, so
# Isaac-GR00T's own setup script (setup_RoboCasa.sh) creates a dedicated venv
# for it and downloads ~5GB of kitchen assets. That script unconditionally
# does `rm -rf robocasa_uv` then rebuilds it from scratch on every
# invocation, so the persistent cache directory (bind-mounted by the caller,
# e.g. slurm/eval_groot_robocasa.sh, at $ROBOCASA_PERSIST_DIR) must NOT be
# mounted anywhere under robocasa_uv itself — rm -rf on an active mount
# point fails with "Device or resource busy" (confirmed on a real run).
# Instead: build normally into the ephemeral container path on a cache miss,
# then copy the result into the persistent cache; on a cache hit, symlink
# the persistent copies into place and skip rebuilding entirely.
#
# Usage:
#   scripts/setup_robocasa_env.sh
#
# Re-running is safe: it restores from cache once the completion marker exists.
set -euo pipefail

GR00T_ROOT="${GR00T_ROOT:-/workspace/Isaac-GR00T}"
ROBOCASA_UV="${GR00T_ROOT}/gr00t/eval/sim/robocasa/robocasa_uv"
ROBOCASA_VENV="${ROBOCASA_UV}/.venv"
ROBOCASA_REPO="${GR00T_ROOT}/external_dependencies/robocasa/robocasa"

PERSIST_DIR="${ROBOCASA_PERSIST_DIR:-/opt/robocasa_persist}"
PERSIST_VENV="${PERSIST_DIR}/venv"
PERSIST_ASSETS="${PERSIST_DIR}/assets"
MARKER="${PERSIST_DIR}/.setup_complete"

# Directories download_kitchen_assets.py actually populates under
# $ROBOCASA_REPO (relative paths; see its DOWNLOAD_ASSET_REGISTRY). Note
# aigen_objs is deliberately skipped by that script ("too large to download
# initially"), so it's not included here.
ASSET_DIRS=(
    "models/assets/textures"
    "models/assets/fixtures"
    "models/assets/objects/objaverse"
    "models/assets/generative_textures"
)

if [ -f "$MARKER" ]; then
    echo "RoboCasa environment already cached at $PERSIST_DIR — restoring symlinks."
    mkdir -p "$ROBOCASA_UV"
    ln -sfn "$PERSIST_VENV" "$ROBOCASA_VENV"
    for rel in "${ASSET_DIRS[@]}"; do
        mkdir -p "$(dirname "${ROBOCASA_REPO}/${rel}")"
        ln -sfn "${PERSIST_ASSETS}/${rel}" "${ROBOCASA_REPO}/${rel}"
    done
    echo "RoboCasa environment ready (restored from cache) → $ROBOCASA_VENV"
    exit 0
fi

cd "$GR00T_ROOT"

echo "Setting up RoboCasa venv (robosuite + robocasa)..."
# setup_RoboCasa.sh already installs everything and downloads the kitchen
# assets itself (echo y | python download_kitchen_assets.py, with macro setup
# as a side effect of that) — do NOT re-run either step here. A second,
# unpiped call to download_kitchen_assets crashes with EOFError when run
# non-interactively (input() has no stdin under apptainer exec).
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh

echo "Caching venv + downloaded assets to persistent storage ($PERSIST_DIR)..."
mkdir -p "$PERSIST_DIR" "$PERSIST_ASSETS"
rm -rf "$PERSIST_VENV"
cp -a "$ROBOCASA_VENV" "$PERSIST_VENV"
for rel in "${ASSET_DIRS[@]}"; do
    src="${ROBOCASA_REPO}/${rel}"
    dst="${PERSIST_ASSETS}/${rel}"
    if [ -d "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        rm -rf "$dst"
        cp -a "$src" "$dst"
    fi
done

touch "$MARKER"
echo "RoboCasa environment ready → $ROBOCASA_VENV (cached to $PERSIST_DIR for future runs)"
