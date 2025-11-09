# Training Guide - Ranjana OCR Model

## Quick Start

```bash
cd python-model
python train.py
```

This will:
- Automatically check and prepare dataset if needed
- Convert labels to Ranjana if needed
- Train for 500 epochs (default)
- Resume from latest checkpoint if available
- Save checkpoints every 5 epochs

## Features

✅ **All Improvements Included:**
- Dual learning rate schedulers (CosineAnnealing + ReduceLROnPlateau)
- Detailed progress tracking (improvements, trends, loss)
- Advanced data augmentation
- Improved architecture with attention
- Automatic checkpoint resume
- Dataset preparation and label conversion

## Usage

### Basic Training (500 epochs)

```bash
python train.py
```

### Custom Epochs

```bash
python train.py --epochs 300
```

### Resume from Latest Checkpoint

```bash
python train.py --resume_latest
```

### Resume from Specific Checkpoint

```bash
python train.py --resume checkpoints/epoch_0200.pth
```

### Prepare Dataset First

```bash
python train.py --prepare_dataset --convert_labels
```

### Custom Configuration

```bash
python train.py --epochs 500 --batch_size 64 --lr 0.001 --checkpoint_interval 10
```

## Command Line Options

- `--epochs`: Number of training epochs (default: 500)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--prepare_dataset`: Prepare combined dataset before training
- `--convert_labels`: Convert labels to Ranjana before training
- `--checkpoint_interval`: Save periodic checkpoint every N epochs (default: 5)
- `--resume`: Resume from specific checkpoint
- `--resume_latest`: Automatically resume from latest checkpoint
- `--auto_setup`: Automatically check and setup dataset/labels (default: True)

## Training Progress

The training script shows:
- ✨ Improvement messages (even small ones)
- 📈 Trend direction (UP/DOWN/STABLE)
- Loss improvements
- Learning rate changes
- Recent accuracy range
- Epochs since last improvement

## Output Files

- `best_character_crnn_improved.pth`: Best model (saved when accuracy improves)
- `character_crnn_improved_final.pth`: Final model (saved at end of training)
- `checkpoints/epoch_XXXX.pth`: Periodic checkpoints
- `checkpoints/best_model.pth`: Best model (in checkpoints directory)
- `checkpoints/final_model.pth`: Final model (in checkpoints directory)
- `training_curves_improved.png`: Training curves visualization

## Expected Results

- **Validation Accuracy**: 99.3-99.6% (after 500 epochs)
- **Dataset Image Recognition**: >99%
- **Real-World Performance**: 88-92%

## Troubleshooting

### Dataset Not Found
```bash
python train.py --prepare_dataset
```

### Labels Not Converted
```bash
python train.py --convert_labels
```

### Out of Memory
```bash
python train.py --batch_size 32
```

### Resume Training
```bash
python train.py --resume_latest
```

## Notes

- Training will automatically resume from latest checkpoint if available
- Checkpoints are saved every 5 epochs by default
- Best model is saved whenever validation accuracy improves
- Final model is saved at the end of training
- All improvements are included by default

