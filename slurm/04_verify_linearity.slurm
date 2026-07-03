#!/bin/bash
# Single job: verify linearity hypothesis and save factorizations.
# Must run AFTER 03_extract_vectors.py completes.
#
# Usage:
#   sbatch slurm/04_verify_linearity.slurm

#SBATCH --job-name=compsteer_linearity
#SBATCH --output=logs/linearity_%j.out
#SBATCH --error=logs/linearity_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source ~/.bashrc
conda activate compsteer
cd /home/hzhang/work/latentSteerVLA

mkdir -p logs results/linearity

python scripts/04_verify_linearity.py \
    --library_root  steering_library \
    --results_root  results/linearity \
    --ranks         4 8 16 32 \
    --methods       svd nmf \
    --device        cuda

echo "Linearity verification complete."
