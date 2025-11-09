# 🚀 How to Start All Services

This guide explains how to start the Python OCR service, Java backend, and React frontend in the correct order.

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Python 3.8+ with required packages (see `python-model/requirements.txt`)
- ✅ Java 17+ installed
- ✅ Maven 3.6+ installed
- ✅ Node.js 16+ and npm installed
- ✅ Trained model file: `python-model/best_character_crnn_improved.pth`

## 🔍 Step 1: Verify Model

First, verify that your trained model is correct:

```powershell
cd python-model
python test_trained_model.py
```

You should see:
- `[SUCCESS] Model contains Devanagari characters - READY FOR USE!`
- Validation accuracy (should be high, e.g., 99%+)

## 🐍 Step 2: Start Python OCR Service

**Terminal 1: Python OCR Service (Port 5000)**

```powershell
cd python-model
python ocr_service_ar.py
```

Expected output:
```
[INFO] EasyOCR available for enhanced character detection
Loading ImprovedCharacterCRNN model from best_character_crnn_improved.pth...
[OK] Model loaded successfully! Type: ImprovedCharacterCRNN, Characters: 74
[INFO] Character set: 8 ASCII, 66 Unicode (Ranjana)
 * Running on http://127.0.0.1:5000
```

**Verify it's working:**
- Open browser: http://localhost:5000/health
- Should return: `{"status": "healthy", "model_loaded": true, ...}`

## ☕ Step 3: Start Java Backend

**Terminal 2: Java Backend (Port 8080)**

```powershell
cd javabackend
mvn spring-boot:run
```

Or use the PowerShell script:
```powershell
.\start_backend.ps1
```

Expected output:
```
Started LipikaApplication in X.XXX seconds
Tomcat started on port(s): 8080 (http)
```

**Verify it's working:**
- Open browser: http://localhost:8080/api/health
- Should return: `{"success": true, "data": {"status": "UP", ...}}`

## ⚛️ Step 4: Start React Frontend

**Terminal 3: React Frontend (Port 3000)**

```powershell
cd frontend
npm install
npm run dev
```

Expected output:
```
  VITE v5.x.x  ready in XXX ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

**Verify it's working:**
- Open browser: http://localhost:3000
- Should see the Lipika OCR application UI

## ✅ Verification Checklist

After starting all services, verify:

1. **Python OCR Service:**
   - ✅ http://localhost:5000/health returns `model_loaded: true`
   - ✅ Model contains Devanagari characters

2. **Java Backend:**
   - ✅ http://localhost:8080/api/health returns `status: UP`
   - ✅ Can connect to Python service at http://localhost:5000

3. **React Frontend:**
   - ✅ http://localhost:3000 loads the UI
   - ✅ Can upload images and see OCR results
   - ✅ Translations work (Ranjana → Devanagari → English)

## 🔄 Service Order

**Important:** Start services in this order:
1. **Python OCR Service** (must be running first)
2. **Java Backend** (depends on Python service)
3. **React Frontend** (depends on Java backend)

## 🛑 Stopping Services

To stop services:
- **Python:** Press `Ctrl+C` in Terminal 1
- **Java:** Press `Ctrl+C` in Terminal 2
- **Frontend:** Press `Ctrl+C` in Terminal 3

## 🐛 Troubleshooting

### Python Service Issues

**Problem:** Model not found
```
[ERROR] No character model found
```
**Solution:** Ensure `best_character_crnn_improved.pth` exists in `python-model/` directory

**Problem:** Port 5000 already in use
```
Address already in use
```
**Solution:** 
- Kill the process using port 5000: `netstat -ano | findstr :5000`
- Or change port in `ocr_service_ar.py`: `app.run(host='0.0.0.0', port=5001)`

### Java Backend Issues

**Problem:** Cannot connect to Python service
```
Connection refused to http://localhost:5000
```
**Solution:** Ensure Python OCR service is running first

**Problem:** Port 8080 already in use
```
Port 8080 was already in use
```
**Solution:** 
- Change port in `javabackend/src/main/resources/application.properties`: `server.port=8081`
- Update frontend API URL if needed

### Frontend Issues

**Problem:** Cannot connect to backend
```
Network Error: Failed to fetch
```
**Solution:** 
- Verify Java backend is running on http://localhost:8080
- Check `frontend/src/services/ocrService.js` for correct API URL
- Check browser console for CORS errors

**Problem:** npm install fails
```
npm ERR! ...
```
**Solution:** 
- Delete `node_modules` and `package-lock.json`
- Run `npm install` again
- If issues persist, try `npm cache clean --force`

## 📝 Quick Start Script (PowerShell)

You can also create a script to start all services:

```powershell
# start_all.ps1
Write-Host "Starting Lipika OCR System..." -ForegroundColor Cyan

# Start Python service (in background)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd python-model; python ocr_service_ar.py"

# Wait a bit for Python to start
Start-Sleep -Seconds 5

# Start Java backend (in background)
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd javabackend; mvn spring-boot:run"

# Wait a bit for Java to start
Start-Sleep -Seconds 10

# Start Frontend
cd frontend
npm run dev
```

## 🎯 Testing the Full Flow

1. **Upload an image** with Ranjana/Devanagari text
2. **OCR should detect** characters and return Devanagari text
3. **Translation should work** automatically (Ranjana → Devanagari)
4. **English translation** should be available via API (Devanagari → English)

## 📊 Model Information

- **Model Type:** ImprovedCharacterCRNN
- **Characters:** 74 total (66 Devanagari, 8 ASCII)
- **Validation Accuracy:** ~99% (check with `test_trained_model.py`)
- **Training Epochs:** 200 (or as specified during training)

## 🔗 Service URLs

- **Python OCR Service:** http://localhost:5000
- **Java Backend:** http://localhost:8080
- **React Frontend:** http://localhost:3000

## 📚 Additional Resources

- Model testing: `python-model/test_trained_model.py`
- Training instructions: `python-model/RETRAIN_INSTRUCTIONS.md`
- API documentation: http://localhost:5000 (Python service)
- Backend API docs: http://localhost:8080/api/health

