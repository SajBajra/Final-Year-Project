# Python Model - Ranjana OCR Service

## Overview
This folder contains the core OCR functionality - the machine learning model and REST API service for Ranjana script recognition.

## Files
- `ocr_service.py` - Main OCR API service (PRODUCTION)
- `app.py` - Legacy Flask web app
- `train_crnn_enhanced.py` - Model training script
- `cli.py` - Command-line interface
- `generate_dataset_and_labels.py` - Dataset generation
- `requirements.txt` - Python dependencies
- `enhanced_chars.txt` - Supported character set
- `templates/` - Web UI templates

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run OCR Service
```bash
python ocr_service.py
```

### Train Model
```bash
python cli.py train --data dataset --epochs 100
```

### API Endpoints
- `GET /health` - Health check
- `POST /predict` - OCR prediction (multipart)
- `POST /predict/base64` - OCR prediction (base64)

## Model Architecture
- **EnhancedCRNN**: CNN feature extractor + bidirectional LSTM
- **CTC Decoder**: Connectionist Temporal Classification
- **Beam Search**: Advanced decoding algorithm

## Output
Returns JSON with recognized Ranjana text from input images.

---

**Note**: Model checkpoints (*.pth files) are not included in this repo. Train your own or contact maintainer.

