#!/usr/bin/env bash
#
# Fine-tune π0.5 (pi05) on our LeRobot v3.0 dataset, inside the
# compsteer-pi05 image (see Dockerfile.pi05 / docker-compose.yml).
#
# The dataset (lucazanett/circular_obj_30fps) ships quantile stats, so pi05's
# default quantile normalization works without overrides.
#
# Usage:
#   ./finetune_pi05.sh                            # defaults: LoRA, 30k steps, batch 16
#   LORA=false BATCH_SIZE=8 ./finetune_pi05.sh    # full fine-tune of all 4B params
#   LORA_R=32 LORA_ALPHA=64 ./finetune_pi05.sh    # bigger adapters
#   STEPS=5000 BATCH_SIZE=4 ./finetune_pi05.sh    # quick run / smaller GPU
#   RESUME=true ./finetune_pi05.sh                # continue from last checkpoint
#   WANDB=true ./finetune_pi05.sh                 # log to wandb (needs WANDB_API_KEY)
#   ./finetune_pi05.sh --policy.train_expert_only=true   # extra lerobot-train flags
#
# LoRA (default) uses lerobot's built-in PEFT integration with pi05's default
# targets: LoRA adapters on the action expert's q/v projections + full training
# of the action/state projection layers; the PaliGemma VLM stays frozen.
# Checkpoints then contain only adapter weights (MBs instead of ~23 GB).
#
# Memory notes (local RTX 6000 Ada, 48 GB): full fine-tune needs ~36 GB at
# batch 8 with gradient checkpointing; LoRA is far lighter, allowing batch 16+.
# If you OOM, lower BATCH_SIZE.
#
# Checkpoints land on the host under outputs/pi05/<JOB_NAME>/checkpoints/
# (the service mounts ./outputs/pi05 at /workspace/outputs).

set -euo pipefail
cd "$(dirname "$0")"

DATASET="${DATASET:-lucazanett/circular_obj_30fps}"
JOB_NAME="${JOB_NAME:-pi05_circular_obj}"
PRETRAINED="${PRETRAINED:-lerobot/pi05_base}"
STEPS="${STEPS:-30000}"
LORA="${LORA:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
if [[ "${LORA}" == "true" ]]; then
    BATCH_SIZE="${BATCH_SIZE:-16}"
else
    BATCH_SIZE="${BATCH_SIZE:-8}"
fi
SAVE_FREQ="${SAVE_FREQ:-5000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WANDB="${WANDB:-false}"
RESUME="${RESUME:-false}"

PEFT_ARGS=()
if [[ "${LORA}" == "true" ]]; then
    PEFT_ARGS=(
        --peft.method_type=LORA
        --peft.r="${LORA_R}"
        --peft.lora_alpha="${LORA_ALPHA}"
    )
fi

# Container-side path; maps to ./outputs/pi05/${JOB_NAME} on the host.
OUTPUT_DIR="/workspace/outputs/${JOB_NAME}"

if [[ "${RESUME}" == "true" ]]; then
    exec docker compose run --rm pi05-finetune lerobot-train \
        --resume=true \
        --config_path="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json" \
        "$@"
fi

exec docker compose run --rm pi05-finetune lerobot-train \
    --dataset.repo_id="${DATASET}" \
    --policy.type=pi05 \
    --policy.pretrained_path="${PRETRAINED}" \
    --policy.push_to_hub=false \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --num_workers="${NUM_WORKERS}" \
    --wandb.enable="${WANDB}" \
    "${PEFT_ARGS[@]}" \
    "$@"
