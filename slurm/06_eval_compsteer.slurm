#!/bin/bash
# Single job: evaluate CompSteer on all four splits.
# Runs on one GPU — uses ManiSkill GPU-parallel envs for speed.
# Must run AFTER 04_verify_linearity.slurm and 05_train_encoders.py.
#
# For the ablation sweep, submit this script multiple times with different args:
#   sbatch slurm/06_eval_compsteer.slurm groot svd 16 cosine
#   sbatch slurm/06_eval_compsteer.slurm groot svd 4  cosine
#   sbatch slurm/06_eval_compsteer.slurm rdt1b svd 16 cosine
#
# Args (positional via $1 ... $4):
#   $1 backbone         default: groot
#   $2 factorize_method default: svd
#   $3 rank             default: 16
#   $4 schedule         default: cosine

#SBATCH --job-name=compsteer_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source ~/.bashrc
conda activate compsteer
cd /home/hzhang/work/latentSteerVLA

BACKBONE=${1:-groot}
FACTORIZE_METHOD=${2:-svd}
RANK=${3:-16}
SCHEDULE=${4:-cosine}

# Map backbone to model path
if [ "$BACKBONE" = "groot" ]; then
    MODEL_PATH="nvidia/GR00T-N1-2B"
elif [ "$BACKBONE" = "groot_1b" ]; then
    MODEL_PATH="nvidia/GR00T-N1-1B"
elif [ "$BACKBONE" = "rdt1b" ]; then
    MODEL_PATH="robotics-diffusion-transformer/rdt-1b"
else
    echo "Unknown backbone: $BACKBONE"
    exit 1
fi

echo "======================================"
echo "Evaluating CompSteer"
echo "  Backbone:          $BACKBONE"
echo "  Model:             $MODEL_PATH"
echo "  Factorize method:  $FACTORIZE_METHOD"
echo "  Rank:              $RANK"
echo "  Schedule:          $SCHEDULE"
echo "======================================"

mkdir -p logs results/eval

python scripts/06_eval_compsteer.py \
    --backbone          "$BACKBONE" \
    --model_path        "$MODEL_PATH" \
    --library_root      steering_library \
    --rank              "$RANK" \
    --factorize_method  "$FACTORIZE_METHOD" \
    --splits            A B C D \
    --num_episodes      50 \
    --num_envs          8 \
    --injection_mode    hidden \
    --schedule          "$SCHEDULE" \
    --schedule_peak     1.0 \
    --alpha             1.0 \
    --beta              1.0 \
    --results_root      results/eval \
    --device            cuda

echo "Evaluation complete."
