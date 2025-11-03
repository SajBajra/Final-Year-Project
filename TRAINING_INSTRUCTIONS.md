# How to Train the Character Model for Lipika

## Quick Start

### Step 1: Navigate to Python Model Directory

```bash
cd python-model
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- torchvision
- flask
- flask-cors
- pillow
- opencv-python
- numpy
- tqdm
- matplotlib

### Step 3: Train the Model

```bash
python train_character_crnn.py \
  --images ../char_dataset/images \
  --train_labels ../char_dataset/train_labels.txt \
  --val_labels ../char_dataset/val_labels.txt \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001
```

### Parameters:

- `--images`: Path to character images folder (default: ../char_dataset/images)
- `--train_labels`: Training labels file (default: ../char_dataset/train_labels.txt)
- `--val_labels`: Validation labels file (default: ../char_dataset/val_labels.txt)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)

### Step 4: Wait for Training

Training typically takes:
- **CPU**: 6-12 hours for 100 epochs
- **GPU**: 1-2 hours for 100 epochs

### Step 5: Model Output

The model will be saved as:
- `best_character_crnn.pth` - Best model based on validation accuracy

**Training curves** saved to:
- `training_curves.png`

## What the Model Does

The model is a **CharacterCRNN** that:
1. Takes 64×64 grayscale character images
2. Uses CNN to extract features
3. Uses LSTM for sequence modeling
4. Predicts individual Ranjana characters

## Expected Results

With 164K training images and 82 character classes:

- **Training Accuracy**: 95-99%
- **Validation Accuracy**: 90-95%
- **Best Model**: Automatically saved when validation accuracy improves

## After Training

### Use the Model

```bash
python ocr_service_ar.py
```

The service will automatically load `best_character_crnn.pth` on start.

### Test with Frontend

```bash
# Terminal 1: Start OCR service
cd python-model
python ocr_service_ar.py

# Terminal 2: Start frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 and test!

## Dataset Structure

```
char_dataset/
├── images/
│   ├── char_000001.png
│   ├── char_000002.png
│   └── ... (164,000 images)
├── train_labels.txt        # 131,200 samples
└── val_labels.txt          # 32,800 samples
```

### Label Format

```
char_000001.png|अ
char_000002.png|आ
char_000003.png|इ
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python train_character_crnn.py --batch_size 32
```

### Slow Training

Use GPU if available:
- PyTorch automatically detects CUDA
- Install CUDA-enabled PyTorch if needed

### Model Not Improving

1. Check dataset quality
2. Increase training epochs
3. Adjust learning rate
4. Check for class imbalance

## Performance Tips

1. **Use GPU** - Training is 5-10x faster
2. **More Epochs** - Better accuracy with 100+ epochs
3. **Data Augmentation** - Already included in training
4. **Early Stopping** - Model saves best checkpoint automatically

## Next Steps

After training completes:
1. Model is ready for production
2. `.pth` files are gitignored (won't upload to GitHub)
3. Test with real Ranjana images
4. Deploy OCR service
5. Build Java backend (optional)

---

**🎯 You're training a character-level model for Google Lens-style AR OCR!**

