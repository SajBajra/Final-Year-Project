#!/bin/bash

# Bash script to start extended training
# Automatically detects if checkpoint exists and resumes if available

echo "=== Lipika Extended Training ==="
echo ""

# Navigate to python-model directory
cd "$(dirname "$0")"

# Check for existing checkpoint
CHECKPOINT_PATH="best_character_crnn_improved.pth"
RESUME_FLAG=""

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "[OK] Found existing checkpoint: $CHECKPOINT_PATH"
    
    echo ""
    echo "Checking checkpoint information..."
    python3 -c "
import torch
checkpoint = torch.load('best_character_crnn_improved.pth', map_location='cpu')
epoch = checkpoint.get('epoch', 'N/A')
val_acc = checkpoint.get('val_acc', 'N/A')
print(f'Last epoch: {epoch}')
print(f'Best validation accuracy: {val_acc:.2f}%')
" 2>/dev/null || echo "[WARN] Could not read checkpoint info"
    
    echo ""
    echo "[INFO] You can resume from this checkpoint or train fresh."
    read -p "Resume from checkpoint? (Y/n): " choice
    
    if [ -z "$choice" ] || [ "${choice,,}" = "y" ]; then
        RESUME_FLAG="--resume $CHECKPOINT_PATH"
        echo "[OK] Will resume from checkpoint"
    else
        echo "[INFO] Will start fresh training (existing checkpoint will be overwritten)"
    fi
else
    echo "[INFO] No existing checkpoint found. Starting fresh training."
fi

# Training parameters
read -p "Enter number of epochs (default: 250): " EPOCHS
EPOCHS=${EPOCHS:-250}

read -p "Enter batch size (default: 64): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-64}

read -p "Enter learning rate (default: 0.001, use 0.0001 for fine-tuning): " LR
LR=${LR:-0.001}

# Dataset paths
IMAGES_PATH="../prepared_dataset/images"
TRAIN_LABELS_PATH="../prepared_dataset/train_labels.txt"
VAL_LABELS_PATH="../prepared_dataset/val_labels.txt"

# Check if dataset exists
if [ ! -d "$IMAGES_PATH" ]; then
    echo "[ERROR] Dataset not found at: $IMAGES_PATH"
    echo "Please make sure you have prepared the dataset first."
    exit 1
fi

echo ""
echo "=== Training Configuration ==="
echo "Images: $IMAGES_PATH"
echo "Train labels: $TRAIN_LABELS_PATH"
echo "Val labels: $VAL_LABELS_PATH"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
if [ -n "$RESUME_FLAG" ]; then
    echo "Resume: Yes (from $CHECKPOINT_PATH)"
else
    echo "Resume: No (fresh training)"
fi
echo ""

read -p "Start training? (Y/n): " confirm
if [ -n "$confirm" ] && [ "${confirm,,}" != "y" ]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting training..."
echo "Press Ctrl+C to stop training"
echo ""

# Build and execute command
CMD="python3 train_character_crnn_improved.py --images $IMAGES_PATH --train_labels $TRAIN_LABELS_PATH --val_labels $VAL_LABELS_PATH --epochs $EPOCHS --batch_size $BATCH_SIZE --lr $LR"

if [ -n "$RESUME_FLAG" ]; then
    CMD="$CMD $RESUME_FLAG"
fi

eval $CMD
