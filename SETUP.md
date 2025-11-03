# Setup Instructions

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Java 17+ (for backend)
- Node.js 18+ (for frontend)

## Step 1: Python OCR Service

```bash
# Navigate to Python model directory
cd python-model

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python ocr_service.py
```

## Step 2: Train Your First Model

```bash
cd python-model

# Generate training dataset
python generate_dataset_and_labels.py --variations 10

# Train model
python cli.py train --data dataset --epochs 50

# Model saved as: enhanced_crnn_model.pth
```

## Step 3: Test the API

```bash
# Start service
cd python-model
python ocr_service.py

# In another terminal, test it
curl -X POST http://localhost:5000/predict -F "image=@test_image.png"
```

## Step 4: Build Java Backend (Future)

```bash
cd javabackend

# Create Spring Boot project (follow javabackend/README.md)
mvn spring-boot:run
```

## Step 5: Build React Frontend (Future)

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

## Docker Setup (Optional)

```dockerfile
# Dockerfile for Python service
FROM python:3.9-slim
WORKDIR /app
COPY python-model/requirements.txt .
RUN pip install -r requirements.txt
COPY python-model/ .
CMD ["python", "ocr_service.py"]
```

## Troubleshooting

### Model not loading
- Ensure `enhanced_crnn_model.pth` exists in `python-model/` directory
- Train a model first: `python cli.py train`

### Port already in use
- Change port: `python ocr_service.py` (edit default port in code)
- Or: `python cli.py web --port 8000`

### CORS errors
- Ensure `flask-cors` is installed
- Check frontend is calling correct URL

### GPU issues
- Install CUDA-compatible PyTorch
- Service works on CPU but slower

## Next Steps

1. Train a better model with more data
2. Deploy Python service to cloud (AWS/GCP)
3. Build Java Spring Boot backend
4. Build React frontend with AR
5. Add translation features

