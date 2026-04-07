#!/bin/bash

# Muon & RNNP Training Pipeline Scripts
# Standardized script for 60M, 130M, 350M models with both optimizers

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Auto-load environment variables from .env if present.
ENV_FILE="$PROJECT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# Default parameters
MODEL_SIZE="60m"
OPTIMIZER="muon"
NUM_GPUS="4"
LEARNING_RATE="0.001"
LR_MATRIX=""
LR_ADAM=""
NUM_STEPS="20000"
BATCH_SIZE="32"
TOTAL_BATCH_SIZE="512"
WARMUP_STEPS="2000"
WEIGHT_DECAY="1e-5"
SAVE_EVERY="10000"
EVAL_EVERY="1000"
WANDB_PROJECT="${WANDB_PROJECT:-llama-pretraining}"
WANDB_NAME=""
R_VAL="1.833"

# Default save directory with timestamp
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
SAVE_DIR="$PROJECT_DIR/checkpoints/llama_${MODEL_SIZE}-${TIMESTAMP}-${OPTIMIZER}"

# Function to display help
show_help() {
    cat << EOF
Muon & RNNP Training Pipeline

Usage: $0 [OPTIONS]

Required parameters:
  --model_size    Model size: 60m, 130m, 350m, 1b
  --optimizer     Optimizer: muon, RMNP

Hardware:
  --num_gpus      Number of GPUs (default: 4). Gradient accumulation is
                  auto-computed as total_batch_size / (batch_size * num_gpus)

Basic training parameters:
  --lr            Learning rate (default: 0.001)
  --lr_matrix     Matrix learning rate (required for RMNP)
  --lr_adam       Adam learning rate (required for RMNP)
  --num_steps     Number of training steps (default: 20000)
  --batch_size    Per-GPU batch size (default: 32)
  --total_batch_size  Total batch size across all GPUs (default: 512)
  --warmup_steps  Warmup steps (default: 2000)
  --weight_decay  Weight decay (default: 1e-5)

Monitoring:
Logging and checkpointing:
  --save_every    Save checkpoint every N steps (default: 10000)
  --eval_every    Evaluate every N steps (default: 1000)
  --save_dir      Directory to save checkpoints (default: auto-generated)
  --wandb_name    WandB run name (default: auto-generated)
  --continue_from Directory to continue training from

Examples:
  # Muon 60M model
  $0 --model_size 60m --optimizer muon --lr 0.001

  # RMNP 130M model
  $0 --model_size 130m --optimizer RMNP --lr_matrix 0.003 --lr_adam 0.001

  # RMNP 350M model
  $0 --model_size 350m --optimizer RMNP --lr_matrix 0.003 --lr_adam 0.001
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lr_matrix)
            LR_MATRIX="$2"
            shift 2
            ;;
        --lr_adam)
            LR_ADAM="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --total_batch_size)
            TOTAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --save_every)
            SAVE_EVERY="$2"
            shift 2
            ;;
        --eval_every)
            EVAL_EVERY="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --wandb_name)
            WANDB_NAME="$2"
            shift 2
            ;;
        --r)                 
            R_VAL="$2"       
            shift 2          
            ;;               
        --continue_from)
            CONTINUE_FROM="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ ! "$MODEL_SIZE" =~ ^(60m|130m|135m|350m|1b)$ ]]; then
    echo "Error: --model_size must be one of: 60m, 130m, 135m, 350m, 1b"
    exit 1
fi

if [[ ! "$OPTIMIZER" =~ ^(muon|RMNP|shampoo|soap|new_optimizer)$ ]]; then
    echo "Error: --optimizer must be one of: muon, RMNP, shampoo, soap, new_optimizer"
    exit 1
fi

# Validate lr_matrix and lr_adam for all optimizers
if [[ -z "$LR_MATRIX" || -z "$LR_ADAM" ]]; then
    echo "Error: $OPTIMIZER requires both --lr_matrix and --lr_adam to be specified"
    exit 1
fi

# Auto-compute gradient accumulation: total_batch_size / (batch_size * num_gpus)
GRAD_ACCUM=$(( TOTAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS) ))
if [[ $(( GRAD_ACCUM * BATCH_SIZE * NUM_GPUS )) -ne $TOTAL_BATCH_SIZE ]]; then
    echo "Error: total_batch_size ($TOTAL_BATCH_SIZE) must be divisible by batch_size ($BATCH_SIZE) * num_gpus ($NUM_GPUS) = $(( BATCH_SIZE * NUM_GPUS ))"
    exit 1
fi
if [[ $GRAD_ACCUM -lt 1 ]]; then
    echo "Error: gradient_accumulation=$GRAD_ACCUM < 1. Reduce batch_size or num_gpus, or increase total_batch_size."
    exit 1
fi

# Set model config based on size
case $MODEL_SIZE in
    60m)
        MODEL_CONFIG="$PROJECT_DIR/configs/llama_60m.json"
        ;;
    130m|135m)
        MODEL_CONFIG="$PROJECT_DIR/configs/llama_135m.json"
        ;;
    350m)
        MODEL_CONFIG="$PROJECT_DIR/configs/llama_350m.json"
        ;;
    1b)
        MODEL_CONFIG="$PROJECT_DIR/configs/llama_1b.json"
        ;;
esac

# Generate WandB name if not provided
if [[ -z "$WANDB_NAME" ]]; then
    WANDB_NAME="llama-${MODEL_SIZE}-${OPTIMIZER}-lr${LEARNING_RATE}"
    if [[ "$OPTIMIZER" == "RMNP" || "$OPTIMIZER" == "shampoo" || "$OPTIMIZER" == "soap" || "$OPTIMIZER" == "new_optimizer" ]]; then
        WANDB_NAME="llama-${MODEL_SIZE}-${OPTIMIZER}-matrix${LR_MATRIX}-adam${LR_ADAM}"
    fi
fi

# Update save directory with complete information
SAVE_DIR="$PROJECT_DIR/checkpoints/llama_${MODEL_SIZE}-${TIMESTAMP}-${OPTIMIZER}"
if [[ "$OPTIMIZER" == "RMNP" || "$OPTIMIZER" == "shampoo" || "$OPTIMIZER" == "soap" || "$OPTIMIZER" == "new_optimizer" ]]; then
    SAVE_DIR="$PROJECT_DIR/checkpoints/llama_${MODEL_SIZE}-${TIMESTAMP}-${OPTIMIZER}-matrix${LR_MATRIX}-adam${LR_ADAM}"
else
    SAVE_DIR="$PROJECT_DIR/checkpoints/llama_${MODEL_SIZE}-${TIMESTAMP}-${OPTIMIZER}-lr${LEARNING_RATE}"
fi

# Create save directory
mkdir -p "$SAVE_DIR"

echo "===== Muon & RNNP Training Pipeline ====="
echo "Model Size: $MODEL_SIZE"
echo "Optimizer: $OPTIMIZER"
echo "Num GPUs: $NUM_GPUS"
echo "Learning Rate: $LEARNING_RATE"
if [[ "$OPTIMIZER" == "RMNP" ]]; then
    echo "Matrix LR: $LR_MATRIX"
    echo "Adam LR: $LR_ADAM"
fi
echo "Training Steps: $NUM_STEPS"
echo "Batch Size (per GPU): $BATCH_SIZE"
echo "Total Batch Size: $TOTAL_BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUM"
echo "Effective: $NUM_GPUS GPUs x $BATCH_SIZE batch x $GRAD_ACCUM accum = $TOTAL_BATCH_SIZE"
echo "Save Directory: $SAVE_DIR"
echo "WandB Name: $WANDB_NAME"
echo "Model Config: $MODEL_CONFIG"
echo "==========================================="

# Build torchrun command.
# Use the current Python environment instead of a potentially mismatched global torchrun binary.
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node "$NUM_GPUS")
PYTHON_CMD="$PROJECT_DIR/torchrun_main.py"

# Build arguments
ARGS=(
    --model_config "$MODEL_CONFIG"
    --optimizer "$OPTIMIZER"
    --lr "$LEARNING_RATE"
    --num_training_steps "$NUM_STEPS"
    --batch_size "$BATCH_SIZE"
    --total_batch_size "$TOTAL_BATCH_SIZE"
    --gradient_accumulation "$GRAD_ACCUM"
    --warmup_steps "$WARMUP_STEPS"
    --weight_decay "$WEIGHT_DECAY"
    --save_every "$SAVE_EVERY"
    --eval_every "$EVAL_EVERY"
    --save_dir "$SAVE_DIR"
    --wandb_name "$WANDB_NAME"
    --dtype "bfloat16"
    --max_length 256
    --scheduler "cosine"
    --grad_clipping 1.0
)

# Add lr_matrix/lr_adam if specified
if [[ -n "$LR_MATRIX" ]]; then
    ARGS+=(--lr_matrix "$LR_MATRIX")
fi
if [[ -n "$LR_ADAM" ]]; then
    ARGS+=(--lr_adam "$LR_ADAM")
fi

if [[ "$OPTIMIZER" == "new_optimizer" ]]; then
    ARGS+=(--r "$R_VAL")
fi

# Add continue_from if specified
if [[ -n "$CONTINUE_FROM" ]]; then
    ARGS+=(--continue_from "$CONTINUE_FROM")
fi

# Log the command to a file
echo "$(date): Starting training with command:" > "$SAVE_DIR/training_command.log"
echo "${TORCHRUN_CMD[*]} -- $PYTHON_CMD ${ARGS[*]}" >> "$SAVE_DIR/training_command.log"

# Save script parameters
cat > "$SAVE_DIR/training_params.json" << EOF
{
    "model_size": "$MODEL_SIZE",
    "optimizer": "$OPTIMIZER",
    "learning_rate": "$LEARNING_RATE",
    "lr_matrix": "$LR_MATRIX",
    "lr_adam": "$LR_ADAM",
    "num_gpus": $NUM_GPUS,
    "num_training_steps": $NUM_STEPS,
    "batch_size": $BATCH_SIZE,
    "total_batch_size": $TOTAL_BATCH_SIZE,
    "gradient_accumulation": $GRAD_ACCUM,
    "warmup_steps": $WARMUP_STEPS,
    "weight_decay": "$WEIGHT_DECAY",
    "save_every": $SAVE_EVERY,
    "eval_every": $EVAL_EVERY,
    "save_dir": "$SAVE_DIR",
    "wandb_name": "$WANDB_NAME",
    "model_config": "$MODEL_CONFIG",
    "timestamp": "$TIMESTAMP"
}
EOF

echo "Training parameters saved to: $SAVE_DIR/training_params.json"
echo "Starting training..."

# Execute training
cd "$PROJECT_DIR"
exec "${TORCHRUN_CMD[@]}" -- "$PYTHON_CMD" "${ARGS[@]}"
