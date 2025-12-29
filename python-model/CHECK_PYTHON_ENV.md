# Troubleshooting Python Module Not Found

## Problem: "No module named flask" even though it's installed

This usually means you're using a different Python interpreter than where the packages are installed.

## Quick Diagnosis

### Step 1: Check which Python you're using

```bash
python --version
which python    # Linux/Mac
where python    # Windows
```

### Step 2: Check if Flask is installed for that Python

```bash
python -c "import flask; print(flask.__version__)"
```

If this fails, Flask is NOT installed for the Python you're using.

### Step 3: Check where packages are installed

```bash
python -m pip list
```

This shows all packages installed for the current Python.

### Step 4: Check if you're in a virtual environment

Look for `(venv)` or similar in your terminal prompt. If you're NOT in a venv but packages are installed in one, that's the issue.

## Solutions

### Solution 1: Use the correct Python

If you have multiple Python installations:
- Try `python3` instead of `python`
- Try `py -3` on Windows
- Use the full path to the Python executable

### Solution 2: Install in the correct Python

```bash
# Make sure you're using the right Python
python -m pip install flask flask-cors

# Or install all requirements
python -m pip install -r requirements.txt
```

### Solution 3: Use a virtual environment (Recommended)

```bash
# Create venv
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Now install packages
pip install -r requirements.txt

# Run the service
python ocr_service_ar.py
```

### Solution 4: Check PATH

Make sure the Python you want to use is in your system PATH.

## Common Issues

1. **Multiple Python versions**: Python 2 vs Python 3, or different Python 3.x versions
2. **Virtual environment not activated**: Packages installed in venv but running outside it
3. **System vs User installation**: Packages installed for one user but running as another
4. **IDE using different Python**: Your IDE might be using a different Python than your terminal

## Verify Installation

After installing, verify:
```bash
python -c "import flask; print('Flask OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import cv2; print('OpenCV OK')"
```

