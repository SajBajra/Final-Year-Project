# Retraining Model with Ranjana Characters

## Problem
Your current model was trained on ASCII transliteration labels (like 'a', 'i', 'e') instead of Ranjana Unicode characters (like 'अ', 'इ', 'ए'). This causes the model to predict ASCII characters even when processing Ranjana script images.

## Solution
Retrain the model using Ranjana Unicode labels. The Ranjana labels already exist in your project!

## Quick Start (Recommended)

### Option 1: Using PowerShell Script (Windows)
```powershell
cd python-model
.\RETRAIN_RANJANA.ps1
```

### Option 2: Using Python Script
```bash
cd python-model
python retrain_with_ranjana.py
```

### Option 3: Manual Training
```bash
cd python-model
python train_character_crnn_improved.py \
    --images ../prepared_dataset/images \
    --train_labels ../prepared_dataset/train_labels_ranjana.txt \
    --val_labels ../prepared_dataset/val_labels_ranjana.txt \
    --epochs 150 \
    --batch_size 64 \
    --lr 0.001
```

## What Happens During Training

1. **Model Backup**: Existing model is automatically backed up
2. **Dataset Loading**: Loads images and Ranjana Unicode labels
3. **Training**: Trains for 150 epochs (or as specified)
4. **Model Saving**: Saves best model as `best_character_crnn_improved.pth`

## Training Time

- **CPU**: ~2-4 hours for 150 epochs
- **GPU**: ~30-60 minutes for 150 epochs
- **Recommended**: Use GPU if available for faster training

## After Training

1. **Restart OCR Service**: The service will automatically load the new model
2. **Test**: Upload an image and verify it now predicts Ranjana characters
3. **Verify**: Check logs for `[INFO] Unicode characters available: ...`

## File Structure

```
prepared_dataset/
├── images/                    # Character images (same for both)
├── train_labels.txt          # ASCII transliteration labels (OLD)
├── val_labels.txt             # ASCII transliteration labels (OLD)
├── train_labels_ranjana.txt   # Ranjana Unicode labels (NEW) ✓
└── val_labels_ranjana.txt      # Ranjana Unicode labels (NEW) ✓

python-model/
├── best_character_crnn_improved.pth  # Model file (will be replaced)
└── retrain_with_ranjana.py            # Retraining script ✓
```

## Troubleshooting

### Error: "Ranjana labels not found"
**Solution**: Run `convert_labels_to_ranjana.py` first:
```bash
python convert_labels_to_ranjana.py
```

### Error: "Images folder not found"
**Solution**: Make sure `prepared_dataset/images` exists and contains image files.

### Training is too slow
**Solutions**:
- Reduce epochs: `--epochs 50`
- Increase batch size (if GPU memory allows): `--batch_size 128`
- Use GPU if available

### Model still predicts ASCII
**Solutions**:
1. Verify the training used Ranjana labels (check logs)
2. Check model file was updated (check modification time)
3. Restart OCR service completely
4. Check startup logs show Unicode characters in vocabulary

## Temporary Workaround (While Training)

If you need the OCR system to work immediately while training, you can:

1. **Accept ASCII predictions**: The current code already accepts ASCII characters if confidence > 0.5
2. **Use transliteration mapping**: ASCII predictions are automatically mapped to Ranjana when possible
3. **Display results**: The system will show ASCII characters, which you can manually convert

However, **retraining is the proper solution** for accurate Ranjana OCR.

## Verification

After training, verify the model contains Ranjana characters:

```python
import torch
checkpoint = torch.load('best_character_crnn_improved.pth', map_location='cpu')
chars = checkpoint['chars']
ranjana_chars = [c for c in chars if ord(c) > 127]
print(f"Ranjana characters in model: {len(ranjana_chars)}")
print(f"Sample: {ranjana_chars[:20]}")
```

## Next Steps

1. ✅ Run retraining script
2. ✅ Wait for training to complete
3. ✅ Restart OCR service
4. ✅ Test with Ranjana images
5. ✅ Verify predictions are in Ranjana Unicode

---

**Note**: Training will replace the existing model. A backup is created automatically, but you can also manually backup:
```bash
cp best_character_crnn_improved.pth best_character_crnn_improved.pth.backup
```

