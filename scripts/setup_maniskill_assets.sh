#!/bin/bash
# One-time (idempotent-ish) download of ManiSkill physics assets referenced
# by configs/tasks.yaml (YCB pick tasks, PartNet-Mobility cabinets/faucets).
#
# Usage:
#   scripts/setup_maniskill_assets.sh
set -euo pipefail

GROOT_VENV="${GROOT_VENV:-/workspace/Isaac-GR00T/.venv}"

echo "Downloading ManiSkill assets (ycb, partnet_mobility_cabinet, partnet_mobility_faucet)..."
"${GROOT_VENV}/bin/python" -m mani_skill.utils.download_asset \
    ycb partnet_mobility_cabinet partnet_mobility_faucet -y

echo "ManiSkill assets ready."
