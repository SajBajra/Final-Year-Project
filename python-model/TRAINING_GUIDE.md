# Character Model Training Guide

## Quick Training

### Step 1: Navigate to python-model
```bash
cd python-model
```

### Step 2: Train Character Model
```bash
python train_character_crnn.py \
  --images ../char_dataset/images \
  --train_labels ../char_dataset/train_labels.txt \
  --val_labels ../char_dataset/val_labels.txt \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.001
```

### Step 3: Wait for Training
- **GPU**: ~2-4 hours
- **CPU**: ~8-16 hours
- Checkpoints: `best_character_crnn.pth` will be saved

### Step 4: Test AR OCR Service
```bash
python ocr_service_ar.py
```

Then upload an image to `http://localhost:5000`

---

## Expected Results

After training, you should see:
- Training accuracy: 90-95%
- Validation accuracy: 85-90%
- Model saved as: `best_character_crnn.pth`

---

## API Usage After Training

### Test with curl
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@your_test_image.png"
```

### Response Example
```json
{
  "success": true,
  "text": "नेपाली भाषा",
  "characters": [
    {"character": "न", "confidence": 0.987, "bbox": {"x": 10, "y": 5, "width": 25, "height": 30}},
    {"character": "े", "confidence": 0.954, "bbox": {"x": 35, "y": 5, "width": 15, "height": 30}},
    // ... more
  ],
  "count": 15
}
```

---

**Ready for Google Lens-style AR!**

