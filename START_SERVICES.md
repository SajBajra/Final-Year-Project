# How to Start Lipika Services

## ✅ Model Status

**Trained Model Verified!**
- ✅ Validation Accuracy: **98.81%**
- ✅ Classes: 67 characters
- ✅ Epoch: 79
- ✅ Model size: 67.6 MB
- ✅ Ready for production

---

## 🚀 Start Services

### Step 1: Start OCR Service

Open **Terminal 1**:

```bash
cd python-model
python ocr_service_ar.py
```

You should see:
```
============================================================
AR-Ready Ranjana Script OCR Service
============================================================
✓ Character model loaded with 67 characters
Device: cpu
Service running on http://0.0.0.0:5000
Endpoints:
  GET  /health         - Health check
  POST /predict        - OCR with bounding boxes
  POST /predict/base64 - OCR with bounding boxes (base64)
============================================================
```

**Keep this terminal open!** The service needs to stay running.

---

### Step 2: Start Frontend

Open **Terminal 2** (new terminal):

```bash
cd frontend
npm install  # Only needed first time
npm run dev
```

You should see:
```
  VITE ready in XXX ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

---

### Step 3: Test in Browser

1. Open http://localhost:3000 in your browser
2. You should see the **Lipika** interface

**Features to test:**
- 📁 Upload a Ranjana image (drag & drop)
- 📷 Use camera capture
- 🔍 Click "Show AR Overlay"
- 👓 See bounding boxes on characters

---

## 🧪 Quick Test

### Test 1: Health Check

In **Terminal 3** (new terminal):

```bash
curl http://localhost:5000/health
```

Expected output:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "chars_count": 67
}
```

### Test 2: OCR Prediction

You can test with a POST request:

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@test_image.png"
```

---

## 🛠️ Troubleshooting

### "Module not found: flask_cors"

Install missing dependencies:

```bash
cd python-model
pip install flask flask-cors opencv-python
```

### "npm not found"

Install Node.js: https://nodejs.org/

### Frontend can't connect to OCR service

Check:
1. OCR service is running on port 5000
2. No firewall blocking connections
3. Both services in correct directories

### Model not loading

Verify file exists:
```bash
cd python-model
dir best_character_crnn.pth
```

Should show: 67,699,310 bytes (~67 MB)

---

## 📊 What to Expect

### Upload Flow

1. Upload/capture image
2. Loading spinner appears
3. Recognized text displays
4. Click "Show AR Overlay"
5. Bounding boxes appear on image
6. Hover boxes to see character labels

### Results Format

```json
{
  "success": true,
  "text": "नेपाली",
  "characters": [
    {
      "character": "न",
      "confidence": 0.985,
      "bbox": {"x": 10, "y": 5, "width": 25, "height": 30}
    },
    ...
  ],
  "count": 6
}
```

---

## 🎯 Next Steps

Once services are running:

1. ✅ Test with sample Ranjana images
2. ✅ Try camera capture
3. ✅ Toggle AR overlay
4. 📝 Note any issues
5. 🚀 Deploy to production

---

## 🔗 API Endpoints

### GET /health

Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "chars_count": 67
}
```

### POST /predict

OCR prediction with AR bounding boxes

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "success": true,
  "text": "recognized text",
  "characters": [...],
  "count": 0
}
```

---

## 📝 Service Management

### Stop Services

Press `Ctrl+C` in each terminal to stop services

### Restart Services

Simply run the start commands again

### Background Mode

Windows PowerShell:
```powershell
Start-Process python -ArgumentList "ocr_service_ar.py" -WindowStyle Hidden
```

---

## 🎉 Success!

If everything works:
- ✅ OCR service running
- ✅ Frontend displaying
- ✅ Can upload images
- ✅ AR overlay working

**Congratulations! Lipika is fully operational!** 🎉

