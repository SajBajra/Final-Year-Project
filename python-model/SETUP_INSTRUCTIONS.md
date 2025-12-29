# Python Model Setup Instructions

## Quick Setup for New Deployment

### Step 1: Install Python Dependencies

Navigate to the `python-model` folder and install all required packages:

```bash
cd python-model
pip install -r requirements.txt
```

### Step 2: Verify Installation

Check if Flask and other dependencies are installed:

```bash
python -c "import flask; print('Flask version:', flask.__version__)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Step 3: Run the OCR Service

Start the OCR service:

```bash
python ocr_service_ar.py
```

The service will start on `http://localhost:5000` by default.

---

## Alternative: Using Virtual Environment (Recommended)

### Step 1: Create Virtual Environment

```bash
cd python-model
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Service

```bash
python ocr_service_ar.py
```

---

## Troubleshooting

### Issue: "No module named flask"

**Solution:** Install Flask and other dependencies:
```bash
pip install flask flask-cors
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Issue: "No module named torch"

**Solution:** Install PyTorch:
```bash
pip install torch torchvision torchaudio
```

### Issue: Python version compatibility

**Note:** If using Python 3.13+, EasyOCR may not work due to scipy compatibility. The service will work fine without it using standard OpenCV segmentation.

---

## Required Python Packages

The main dependencies are:
- Flask (web framework)
- Flask-CORS (CORS support)
- PyTorch (deep learning)
- Pillow (image processing)
- OpenCV (computer vision)
- NumPy (numerical operations)

All packages are listed in `requirements.txt`.

