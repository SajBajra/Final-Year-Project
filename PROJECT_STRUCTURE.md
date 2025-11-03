# Project Structure Documentation

## 📋 Overview

This is a cleaned, production-ready Ranjana Script OCR system based on CRNN architecture. The codebase has been refactored from dual models (character + word) to a unified word-based OCR service that follows a clean MVP architecture.

## 🎯 MVP Architecture

```
┌─────────────────────────────────────────────────────────┐
│  View Layer (React)                                     │
│  - Image upload interface                               │
│  - AR visualization                                     │
│  - Translation display                                  │
└─────────────────────────────────────────────────────────┘
                         ↕ HTTP/REST API
┌─────────────────────────────────────────────────────────┐
│  Presenter Layer (Java Spring Boot)                     │
│  - Business logic                                       │
│  - Validation                                          │
│  - Response formatting                                  │
│  - Coordinates frontend ↔ OCR service                   │
└─────────────────────────────────────────────────────────┘
                         ↕ HTTP/REST API
┌─────────────────────────────────────────────────────────┐
│  Model Layer (Python OCR Service)                       │
│  - PyTorch CRNN model                                   │
│  - Image preprocessing                                  │
│  - Text recognition                                     │
│  - Unicode normalization                                │
└─────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
FYP/
├── ocr_service.py           # Main OCR API service (PRODUCTION)
│   ├── EnhancedCRNN         # Neural network model
│   ├── /health              # Health check endpoint
│   ├── /predict             # OCR prediction (multipart)
│   └── /predict/base64      # OCR prediction (base64)
│
├── app.py                   # Legacy Flask web app
│   ├── /                    # Web UI
│   ├── /predict             # Delegates to ocr_service
│   ├── /add_sample          # Add training samples
│   └── /train_user          # Trigger retraining
│
├── train_crnn_enhanced.py   # Model training script
│   └── EnhancedCRNN         # Shared model architecture
│
├── cli.py                   # Command-line interface
│   ├── train                # Train model
│   ├── infer                # Run inference
│   └── web                  # Start web app
│
├── generate_dataset_and_labels.py  # Dataset generation
│
├── templates/               # Web UI templates
│   ├── index.html          # Main web interface
│   └── character_index.html # Character mode UI (legacy)
│
├── requirements.txt         # Python dependencies
├── enhanced_chars.txt       # Supported character set
├── chars.txt               # Basic characters
├── README.md               # Project documentation
└── .gitignore             # Git ignore rules
```

## 🔧 Core Components

### 1. OCR Service (`ocr_service.py`)

**Purpose**: Production-ready REST API for OCR functionality

**Key Features**:
- Flask-based HTTP API
- CORS enabled for frontend integration
- Two prediction endpoints (multipart and base64)
- Health check endpoint
- Model loading and inference
- Error handling and logging

**Endpoints**:
- `GET /health` - Service health and model status
- `POST /predict` - OCR from multipart form data
- `POST /predict/base64` - OCR from base64 JSON

### 2. Training Script (`train_crnn_enhanced.py`)

**Purpose**: Train and fine-tune the CRNN model

**Features**:
- Data augmentation
- CTC loss optimization
- Learning rate scheduling
- Model checkpointing
- Training metrics visualization

### 3. CLI (`cli.py`)

**Purpose**: Unified command-line interface

**Commands**:
- `python cli.py train --data dataset` - Train model
- `python cli.py infer --model X --input Y` - Run inference
- `python cli.py web --port 5000` - Start web app

### 4. Model Architecture (`EnhancedCRNN`)

**Components**:
- **CNN**: 5-layer feature extractor (1→64→128→256→512 channels)
- **RNN**: 3-layer bidirectional LSTM (256 hidden units)
- **CTC**: Connectionist Temporal Classification decoder
- **Beam Search**: Advanced decoding algorithm

**Input**: 32×128 grayscale image
**Output**: Unicode text string

## 🔄 Data Flow

### Training Flow
```
generate_dataset_and_labels.py → dataset/
                                ↓
                        train_crnn_enhanced.py
                                ↓
                        enhanced_crnn_model.pth
```

### Inference Flow
```
Upload Image → ocr_service.py
                  ↓
            Preprocess Image
                  ↓
            EnhancedCRNN Model
                  ↓
            CTC Decode + Beam Search
                  ↓
            Unicode Normalization
                  ↓
            Return JSON Response
```

### API Integration Flow
```
React Frontend
      ↓ (HTTP POST with image)
Java Spring Boot Backend
      ↓ (HTTP POST with image)
Python OCR Service
      ↓ (JSON response with text)
Back to Frontend (via Backend)
```

## 🚫 What Was Removed

### Deleted Files
- `character_web_app.py` - Duplicate character-based web app
- `char_segmentation.py` - Character segmentation utilities
- `character_inference.py` - Character-only inference
- `train_character_crnn.py` - Character-only training
- `generate_character_dataset.py` - Character dataset generation
- `sequence_inference.py` - Character sequence inference
- `archive/` - All archived old code
- Various checkpoint and log files

### Rationale
- **Simplification**: Single unified model is easier to maintain
- **Production Focus**: Removed experimental/legacy code
- **Size Reduction**: Eliminated duplicate functionality
- **Clean Architecture**: Clear separation of concerns

## ✅ What Remains

### Essential Files
- `ocr_service.py` - Production OCR API
- `app.py` - Legacy web app (backward compatibility)
- `train_crnn_enhanced.py` - Training script
- `cli.py` - Command interface
- `generate_dataset_and_labels.py` - Dataset creation
- `requirements.txt` - Dependencies
- Character set files

### Build Artifacts (Gitignored)
- `*.pth` - Model checkpoints
- `checkpoints/` - Training checkpoints
- `char_dataset/` - Generated datasets
- `user_datasets/` - User training data
- `__pycache__/` - Python cache
- Log files and images

## 🔌 Integration Examples

### Java Backend
```java
@Component
public class OCRServiceClient {
    @Autowired
    private RestTemplate restTemplate;
    
    public String recognizeText(MultipartFile image) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("image", image.getResource());
        
        HttpEntity<MultiValueMap<String, Object>> request = 
            new HttpEntity<>(body, headers);
        
        OCRResponse response = restTemplate.postForObject(
            "http://localhost:5000/predict",
            request,
            OCRResponse.class
        );
        
        return response.getText();
    }
}
```

### React Frontend
```javascript
import React, { useState } from 'react';

function OCRInterface() {
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  
  const handleImageUpload = async (file) => {
    const formData = new FormData();
    formData.append('image', file);
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setResult(data.text);
    } catch (error) {
      console.error('OCR failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <input 
        type="file" 
        accept="image/*" 
        onChange={(e) => handleImageUpload(e.target.files[0])}
      />
      {loading && <p>Processing...</p>}
      {result && <p>Result: {result}</p>}
    </div>
  );
}

export default OCRInterface;
```

### Python Client
```python
import requests

def recognize_text(image_path):
    """Call OCR service"""
    url = "http://localhost:5000/predict"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        return response.json()['text']
    else:
        raise Exception(f"OCR failed: {response.text}")
```

## 🚀 Next Steps

1. **Deploy OCR Service**: Containerize with Docker
2. **Add Java Backend**: Implement Spring Boot presenter layer
3. **Build React Frontend**: Create modern web interface
4. **Add AR Support**: Implement augmented reality overlay
5. **Translation Integration**: Add translation API calls
6. **Cloud Deployment**: Deploy to AWS/GCP/Azure

## 📝 Notes

- `.gitignore` ensures large files (models, datasets) aren't tracked
- All models use `.pth` extension (PyTorch format)
- Dataset format: `filename.png|ground_truth_text` (one per line)
- Input images expected to be grayscale or will be converted
- Model trained on 32×128 pixel images with CTC loss
- Maximum sequence length handled by CTC decoder

---

**Status**: Production-ready OCR service, ready for MVP deployment

