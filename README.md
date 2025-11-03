# Lipika - Ranjana Script OCR System

A production-ready OCR (Optical Character Recognition) system for Ranjana script using CRNN (CNN + RNN) deep learning architecture. **Lipika** (लिपिका) recognizes Ranjana text from images with Google Lens-style AR overlay and exposes REST API endpoints for integration with external applications.

## 🎯 MVP Overview

### Architecture (Google Lens-style)

The system follows a three-layer MVP architecture:

1. **Model Layer (Python - OCR Service)** `python-model/`: PyTorch-based CRNN neural network that performs image recognition. Exposes REST API endpoints through Flask.
2. **Presenter Layer (Java - Optional)** `javabackend/`: Spring Boot backend that coordinates between frontend and OCR service, handles business logic and validation.
3. **View Layer (React - Frontend)** `frontend/`: Modern web interface for image upload, OCR results display, and optional AR visualization.

## 📁 Project Structure

```
FYP/
├── python-model/           # OCR service and training
│   ├── ocr_service.py      # Main production API
│   ├── app.py              # Legacy web app
│   ├── train_crnn_enhanced.py
│   ├── cli.py
│   ├── templates/
│   └── requirements.txt
├── javabackend/            # Java Spring Boot (to be implemented)
│   └── README.md
├── frontend/               # React frontend (to be implemented)
│   └── README.md
├── README.md               # This file
└── .gitignore             # Excludes large files
```

## 🛠️ Quick Start

### Python OCR Service

```bash
cd python-model

# Install dependencies
pip install -r requirements.txt

# Start OCR service
python ocr_service.py
```

Service runs on `http://localhost:5000` with endpoints:
- `GET /health` - Health check
- `POST /predict` - OCR prediction (multipart)
- `POST /predict/base64` - OCR prediction (base64 JSON)

### Train Model

```bash
cd python-model
python cli.py train --data dataset --epochs 100
```

### CLI Usage

```bash
cd python-model

# Train
python cli.py train --data dataset

# Infer
python cli.py infer --model enhanced_crnn_model.pth --chars enhanced_chars.txt --input test.png

# Web app
python cli.py web --port 5000
```

## 🔌 API Integration

### Example: Python Client
```python
import requests

def recognize_text(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post('http://localhost:5000/predict', files=files)
    return response.json()['text']
```

### Example: JavaScript (React)
```javascript
const formData = new FormData();
formData.append('image', file);

const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
});

const data = await response.json();
console.log(data.text); // Recognized text
```

## 🏗️ Model Architecture

**EnhancedCRNN**:
- **CNN**: 5-layer feature extractor (64→128→256→512 channels)
- **LSTM**: 3-layer bidirectional (256 hidden units)
- **CTC**: Connectionist Temporal Classification decoder
- **Beam Search**: Advanced decoding algorithm

**Input**: 32×128 grayscale image
**Output**: Unicode Ranjana text

## 📊 Supported Characters

See `python-model/enhanced_chars.txt` for complete character list including:
- Vowels: अ, आ, इ, ई, उ, ऊ, ए, ऐ, ओ, औ
- Consonants: क, ख, ग, घ, etc.
- Diacritical marks: ँ, ं, ः, ा, ि, ी, ु, ू, े, ो
- Numbers: ०-९
- Punctuation: ।, ॥

## 🚀 Next Steps

### To Build Java Backend
See `javabackend/README.md` for setup instructions

### To Build React Frontend
See `frontend/README.md` for setup instructions

## 📝 Development

### Current Status
- ✅ Python OCR service working
- ✅ REST API endpoints functional
- ✅ Model training pipeline
- ⏳ Java backend (to be built)
- ⏳ React frontend (to be built)

## 📄 License

MIT License

## 🙏 Acknowledgments

- PyTorch team
- CTC-based CRNN architecture
- Ranjana font: NithyaRanjanaDU

---

**Built with ❤️ for preserving Ranjana script through technology**
