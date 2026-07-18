#!/bin/bash
#PBS -N pi05_cart_imp_lora
#PBS -q gpu
#PBS -l select=1:ngpus=1:ncpus=8
#PBS -j oe
#PBS -o /home/hzhang/work/latentSteerVLA/logs/pi05_cart_imp_resume.pbs.log
#
# Resume the pi05 LoRA fine-tune on lucazanett/cart_imp_data from the last
# checkpoint, submitted through PBS so the GPU is reserved in the scheduler.
#
#   qsub qsub_resume_pi05_cart_imp.sh
#
# Live training log: logs/pi05_cart_imp_resume.log
# The trap makes qdel kill the docker container too, not just the compose client.

set -euo pipefail
cd /home/hzhang/work/latentSteerVLA

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
eval "$(grep '^export WANDB_API_KEY=' ~/.bashrc)"

CONTAINER=pi05-finetune-pbs
trap 'docker rm -f "$CONTAINER" >/dev/null 2>&1 || true' EXIT TERM INT

docker compose run --rm --name "$CONTAINER" pi05-finetune lerobot-train \
    --resume=true \
    --config_path=/workspace/outputs/pi05_cart_imp_lora/checkpoints/last/pretrained_model/train_config.json \
    2>&1 | tee logs/pi05_cart_imp_resume.log
