#!/bin/bash
# Muon 60M Model Training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 60m \
    --optimizer muon \
    --num_gpus 4 \
    --lr_matrix 0.01 \
    --lr_adam 0.001 \
    --num_steps 10000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 1000 \
    --weight_decay 0.1 \
    --save_every 10000 \
    --eval_every 1000 \
    "$@"
