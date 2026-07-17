#!/usr/bin/env bash
#
# Chain job: wait for the running LoRA fine-tune to finish, then launch the
# full-parameter π0.5 fine-tune ("strongest" config for the 48 GB RTX 6000 Ada).
#
# Config rationale (measured on this GPU):
#   - Full FT static memory ≈ 33 GB (bf16 weights 8.2 + grads 8.2 + AdamW 16.4)
#     plus ~0.5–1 GB activations per sample with gradient checkpointing
#     (smoke test: 31.5 GB @ batch 2) → batch 12 ≈ 39–45 GB. Batch 16 would
#     risk OOM, batch 12 is the sweet spot.
#   - STEPS=20000 @ batch 12 = 240k samples ≈ 15.5 epochs — same total sample
#     budget as the standard 30k×8 full-FT recipe, scaled for the larger batch.
#
# Output goes to outputs/pi05/pi05_circular_obj_full/ — the LoRA checkpoints in
# outputs/pi05/pi05_circular_obj_lora/ are untouched.

set -euo pipefail
cd "$(dirname "$0")"

LORA_DIR="outputs/pi05/pi05_circular_obj_lora"

CID=$(docker ps -q --filter name=pi05-finetune | head -1)
if [[ -n "${CID}" ]]; then
    echo "$(date '+%F %T') waiting for LoRA container ${CID} to finish..."
    docker wait "${CID}"
fi

if [[ ! -e "${LORA_DIR}/checkpoints/last" ]]; then
    echo "ERROR: LoRA run left no final checkpoint in ${LORA_DIR}; not starting full FT." >&2
    exit 1
fi
echo "$(date '+%F %T') LoRA run finished; launching full-parameter fine-tune."

exec env \
    HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
    WANDB_API_KEY="$(grep -oP '(?<=^export WANDB_API_KEY=).*' ~/.bashrc | tail -1)" \
    WANDB=true \
    LORA=false \
    BATCH_SIZE=12 \
    STEPS=20000 \
    JOB_NAME=pi05_circular_obj_full \
    ./finetune_pi05.sh --wandb.project=pi05-circular_obj_30fps
