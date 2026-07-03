#!/bin/bash
# SLURM array job: train VAE for all (embodiment, task) pairs.
# Must run AFTER 01_collect_demos.slurm.
#
# One GPU per pair. With 24 pairs, allocate --array=0-23.
# Each VAE trains for 100 epochs on ~50 demos (~15-30 min on A100).
#
# Usage:
#   sbatch slurm/02_train_vae.slurm
#   sbatch --array=0-5 slurm/02_train_vae.slurm   # first 6 pairs

#SBATCH --job-name=compsteer_vae
#SBATCH --output=logs/vae_%A_%a.out
#SBATCH --error=logs/vae_%A_%a.err
#SBATCH --array=0-23
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source ~/.bashrc
conda activate compsteer
cd /home/hzhang/work/latentSteerVLA

# Same pair table as collect_demos (must stay in sync)
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

read -r EMBODIMENT TASK <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Training VAE: $EMBODIMENT × $TASK"
echo "======================================"

mkdir -p logs

python scripts/02_train_vae.py \
    --embodiment    "$EMBODIMENT" \
    --task          "$TASK" \
    --data_root     data \
    --checkpoint_root checkpoints \
    --latent_dim    256 \
    --hidden_dims   512 512 \
    --beta          1.0 \
    --lr            3e-4 \
    --batch_size    128 \
    --num_epochs    100 \
    --action_horizon 16 \
    --device        cuda

echo "Done: VAE for $EMBODIMENT × $TASK"
