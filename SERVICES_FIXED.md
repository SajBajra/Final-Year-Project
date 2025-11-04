# ✅ Services Fixed - Ready to Run!

## 🔧 What Was Fixed

### Issue Found
The OCR service wasn't starting because **`flask-cors`** module was missing.

### Solution Applied
✅ Installed `flask-cors` and verified all dependencies from `requirements.txt`

---

## 🚀 How to Start Services

### Option 1: Use Startup Script (Recommended)

```powershell
.\start_services.ps1
```

This script will:
- Check for Python and Node.js
- Start OCR service in a new terminal
- Start Frontend in a new terminal
- Show you the URLs

### Option 2: Manual Start (Two Terminals)

#### Terminal 1 - OCR Service:
```powershell
cd python-model
python ocr_service_ar.py
```

**Expected Output:**
```
============================================================
AR-Ready Ranjana Script OCR Service
============================================================
✓ Character model loaded with 82 characters
Device: cpu (or cuda)
Service running on http://0.0.0.0:5000
Endpoints:
  GET  /health         - Health check
  POST /predict        - OCR with bounding boxes
  POST /predict/base64 - OCR with bounding boxes (base64)
============================================================
 * Running on http://0.0.0.0:5000
```

#### Terminal 2 - Frontend:
```powershell
cd frontend
npm install  # Only needed first time
npm run dev
```

**Expected Output:**
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

---

## ✅ Verification

### 1. Check OCR Service
Open browser: http://localhost:5000/health

Should return:
```json
{
  "status": "ok",
  "model_loaded": true,
  "chars_count": 82
}
```

### 2. Check Frontend
Open browser: http://localhost:3000

Should see: Lipika homepage with upload and camera options

---

## ⚠️ Troubleshooting

### If OCR Service Won't Start

**Error: ModuleNotFoundError**
```powershell
# Install missing dependencies
cd python-model
pip install -r requirements.txt
```

**Error: Model file not found**
- Make sure `best_character_crnn.pth` exists in `python-model/` folder
- If missing, train the model first: `python train_character_crnn.py --epochs 100`

### If Frontend Won't Start

**Error: npm not found**
- Install Node.js from: https://nodejs.org/
- Make sure to install the LTS version (18+)
- Restart terminal after installation

**Error: Module not found**
```powershell
cd frontend
npm install
```

### If Port Already in Use

**Port 5000 in use:**
```powershell
# Find process using port 5000
netstat -ano | findstr ":5000"

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

**Port 3000 in use:**
```powershell
# Find process using port 3000
netstat -ano | findstr ":3000"

# Kill the process
taskkill /PID <PID> /F
```

---

## 📋 Current Status

✅ **Dependencies Installed**
- flask-cors ✓
- All Python requirements ✓

✅ **Model Ready**
- `best_character_crnn.pth` exists ✓
- 82 characters loaded ✓

✅ **Services Ready**
- OCR Service: Ready to start ✓
- Frontend: Ready to start (if Node.js installed) ✓

---

## 🎯 Next Steps

1. **Start OCR Service** (Terminal 1)
   ```powershell
   cd python-model
   python ocr_service_ar.py
   ```

2. **Start Frontend** (Terminal 2)
   ```powershell
   cd frontend
   npm install  # First time only
   npm run dev
   ```

3. **Test Everything**
   - Go to http://localhost:3000
   - Upload a Ranjana image
   - See OCR results with AR overlay!

---

## 📝 Notes

- The OCR service needs the trained model (`best_character_crnn.pth`)
- Frontend needs Node.js 18+ installed
- Both services can run simultaneously
- Check browser console (F12) for any frontend errors

---

**🎉 Your services are ready to run!**
