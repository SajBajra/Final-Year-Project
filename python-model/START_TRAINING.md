# 🚀 How to Start Training - Quick Guide

## Simple Steps

### Step 1: Navigate to Python Model Directory

```bash
cd python-model
```

### Step 2: Start Training

```bash
python train_character_crnn_improved.py \
  --images ../prepared_dataset/images \
  --train_labels ../prepared_dataset/train_labels.txt \
  --val_labels ../prepared_dataset/val_labels.txt \
  --epochs 150 \
  --batch_size 64 \
  --lr 0.001
```

### Step 3: Wait and Monitor

You'll see output like:
```
============================================================
LIPIKA - IMPROVED Character CRNN Training
============================================================
Images folder: ../prepared_dataset/images
Train labels: ../prepared_dataset/train_labels.txt
Val labels: ../prepared_dataset/val_labels.txt
Epochs: 150
Batch size: 64
Learning rate: 0.001
============================================================

Loaded XXXX character samples from ../prepared_dataset/train_labels.txt
Character set size: XX characters

Using device: cuda (or cpu)

Improved Model created with XX character classes
Total parameters: X,XXX,XXX
Trainable parameters: X,XXX,XXX

Starting IMPROVED training for 150 epochs...

Epoch 1/150:
  Train Loss: X.XXXX, Train Acc: XX.XX%
  Val Loss: X.XXXX, Val Acc: XX.XX%
  LR: 0.001000
  ✓ Saved best model with val_acc: XX.XX%
```

---

## Alternative: Adjust Parameters

### If You Have Limited Memory (Reduce Batch Size)

```bash
python train_character_crnn_improved.py \
  --images ../prepared_dataset/images \
  --train_labels ../prepared_dataset/train_labels.txt \
  --val_labels ../prepared_dataset/val_labels.txt \
  --epochs 150 \
  --batch_size 32 \
  --lr 0.001
```

### If You Want Faster Training (Fewer Epochs for Testing)

```bash
python train_character_crnn_improved.py \
  --images ../prepared_dataset/images \
  --train_labels ../prepared_dataset/train_labels.txt \
  --val_labels ../prepared_dataset/val_labels.txt \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001
```

---

## Prerequisites Check

Make sure you have:

1. ✅ **Prepared Dataset** (should be in `../prepared_dataset/`)
   ```bash
   # Check if exists
   ls ../prepared_dataset/images
   ls ../prepared_dataset/train_labels.txt
   ls ../prepared_dataset/val_labels.txt
   ```

2. ✅ **Python Dependencies** installed
   ```bash
   pip install torch torchvision pillow numpy tqdm matplotlib
   ```

3. ✅ **Enough Disk Space** (model file will be ~200-500 MB)

---

## What to Expect

- **Training Time**: 2-4 hours (GPU) or 8-16 hours (CPU)
- **Output Files**: 
  - `best_character_crnn_improved.pth` - Trained model
  - `training_curves_improved.png` - Training visualization
- **Final Message**: 
  ```
  ✅ Training complete! Best accuracy: XX.XX%
  📁 Model saved as: best_character_crnn_improved.pth
  ```

---

## Troubleshooting

### Error: "ModuleNotFoundError"
**Fix**: Install missing packages
```bash
pip install torch torchvision pillow numpy tqdm matplotlib
```

### Error: "FileNotFoundError" (dataset files)
**Fix**: Make sure you ran the dataset preparation script first
```bash
cd python-model
python prepare_dataset.py --dataset ../Dataset --output ../prepared_dataset
```

### Error: "Out of Memory"
**Fix**: Reduce batch size
```bash
python train_character_crnn_improved.py --batch_size 32 ...
```

### Training Too Slow?
- Use GPU if available (PyTorch will detect automatically)
- Reduce epochs for testing: `--epochs 50`

---

## That's It! 🎉

Just run the command and let it train. The script will:
- ✅ Load your Google Dataset
- ✅ Apply data augmentation
- ✅ Train the improved CRNN model
- ✅ Save the best model automatically
- ✅ Show progress in real-time

**Training is running in the background now!** Check the terminal output to see progress.
