#!/bin/bash
# SLURM array job: collect ManiSkill demos for all (embodiment, task) pairs
#
# Array index → (embodiment, task) pair via a flat index.
# With 4 train embodiments × 6 train tasks = 24 pairs  → --array=0-23
# Add test embodiments/tasks as needed.
#
# Usage:
#   sbatch slurm/01_collect_demos.slurm
#   sbatch --array=0-5 slurm/01_collect_demos.slurm   # first 6 pairs only

#SBATCH --job-name=compsteer_collect
#SBATCH --output=logs/collect_%A_%a.out
#SBATCH --error=logs/collect_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ── Environment ───────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate compsteer          # update to your env name
cd /home/hzhang/work/latentSteerVLA

# ── Pair lookup table ──────────────────────────────────────────────────────────
# Format: "embodiment task"
# Rows must match --array range.  Add / remove pairs as needed.
PAIRS=(
    "panda       pick_cube"
    "panda       stack_cube"
    "panda       peg_insertion"
    "panda       push_t"
    "panda       open_drawer"
    "panda       pick_ycb"
    "xarm6       pick_cube"
    "xarm6       stack_cube"
    "xarm6       peg_insertion"
    "xarm6       push_t"
    "xarm6       open_drawer"
    "xarm6       pick_ycb"
    "ur5e        pick_cube"
    "ur5e        stack_cube"
    "ur5e        peg_insertion"
    "ur5e        push_t"
    "ur5e        open_drawer"
    "ur5e        pick_ycb"
    "widowx      pick_cube"
    "widowx      stack_cube"
    "widowx      peg_insertion"
    "widowx      push_t"
    "widowx      open_drawer"
    "widowx      pick_ycb"
)

# Parse this task's (embodiment, task) pair
read -r EMBODIMENT TASK <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Embodiment: $EMBODIMENT"
echo "Task:       $TASK"
echo "======================================"

mkdir -p logs

python scripts/01_collect_demos.py \
    --embodiment "$EMBODIMENT" \
    --task       "$TASK" \
    --num_demos  50 \
    --num_envs   8 \
    --seed       42 \
    --data_root  data

echo "Done: $EMBODIMENT x $TASK"
