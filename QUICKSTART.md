# Lipika Quick Start Guide

Get Lipika (Ranjana OCR with AR) running in 5 minutes!

## What is Lipika?

Lipika is a Google Lens-style OCR system that:
- 📸 Recognizes Ranjana script from images
- 🔍 Provides character-level detection
- 👓 Shows AR overlay with bounding boxes
- 🌐 Modern React UI

## Quick Setup

### 1. Train the Model (One-time)

```bash
cd python-model
pip install -r requirements.txt

python train_character_crnn.py \
  --epochs 100 \
  --batch_size 64
```

⏱️ Time: 1-6 hours depending on CPU/GPU

### 2. Start OCR Service

```bash
cd python-model
python ocr_service_ar.py
```

✅ Service running on http://localhost:5000

### 3. Start Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

✅ UI running on http://localhost:3000

### 4. Test It!

1. Go to http://localhost:3000
2. Upload a Ranjana image or use camera
3. Click "Show AR Overlay" to see bounding boxes
4. See recognized text in real-time!

## Project Structure

```
Lipika/
├── python-model/          # AI/ML Layer (PyTorch)
│   ├── ocr_service_ar.py  # OCR API service
│   ├── train_character_crnn.py  # Model training
│   └── requirements.txt
├── frontend/              # View Layer (React)
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/    # UI components
│   └── package.json
├── javabackend/           # Presenter Layer (to build)
└── char_dataset/          # Training data (164K images)
```

## Key Files

- **OCR Service**: `python-model/ocr_service_ar.py`
- **Training**: `python-model/train_character_crnn.py`
- **Frontend**: `frontend/src/App.jsx`
- **Model Output**: `best_character_crnn.pth` (gitignored)

## Architecture

```
User uploads image
    ↓
React Frontend (Tailwind CSS)
    ↓
Python OCR Service (Flask API)
    ↓
Character CRNN Model
    ↓
Returns: text + bounding boxes
    ↓
AR Overlay (Google Lens style)
```

## Troubleshooting

### "Module not found"
```bash
cd python-model
pip install -r requirements.txt
```

### "npm not found"
Install Node.js: https://nodejs.org/

### "No trained model"
Train first: `python train_character_crnn.py --epochs 100`

### Model won't load
Check: `best_character_crnn.pth` exists in `python-model/`

## Next Steps

- ✅ Train model
- ✅ Start services
- ✅ Test upload
- 📝 Add Java backend (optional)
- 🚀 Deploy to production

## Documentation

- 📖 Full README: [README.md](README.md)
- 🎓 Training guide: [TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md)
- 🏗️ Project structure: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

**Made with ❤️ for Ranjana script preservation**

