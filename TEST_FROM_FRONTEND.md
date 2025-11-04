# 🖥️ Testing OCR from Frontend - Complete Guide

## Overview

To test the OCR model from the frontend, you need **3 services running**:

1. **Python OCR Service** (port 5000) - The ML model
2. **Java Backend** (port 8080) - API gateway
3. **React Frontend** (port 5173) - User interface

---

## Step-by-Step Setup

### Step 1: Start Python OCR Service

```bash
cd python-model
python ocr_service_ar.py
```

**Expected output:**
```
============================================================
AR-Ready Ranjana Script OCR Service
============================================================
Loading ImprovedCharacterCRNN model from best_character_crnn_improved.pth...
[OK] Model loaded successfully! Type: ImprovedCharacterCRNN, Characters: 63
Device: cuda
Service running on http://0.0.0.0:5000
Endpoints:
  GET  /health         - Health check
  POST /predict        - OCR with bounding boxes
  POST /predict/base64 - OCR with bounding boxes (base64)
============================================================
```

**Verify it's working:**
- Open browser: http://localhost:5000
- Should see the API documentation page
- Check health: http://localhost:5000/health

---

### Step 2: Start Java Backend

#### Option A: Using Command Line (Maven)

```bash
cd javabackend
mvn spring-boot:run
```

#### Option B: Using Eclipse

1. Open Eclipse
2. Import the `javabackend` project
3. Right-click on `LipikaApplication.java`
4. Run As → Java Application

**Expected output:**
```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::                (v3.x.x)

2024-xx-xx INFO ... Starting LipikaApplication
2024-xx-xx INFO ... Started LipikaApplication in X.XXX seconds
```

**Verify it's working:**
- Check health: http://localhost:8080/api/health
- Should return: `{"success":true,"message":"All services are healthy","data":null}`

---

### Step 3: Start React Frontend

```bash
cd frontend
npm install  # Only needed first time
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

**Verify it's working:**
- Open browser: http://localhost:5173
- Should see the Lipika OCR interface

---

## Testing Process

### Test 1: Upload Image from Dataset

1. **Open frontend**: http://localhost:5173
2. **Click "Upload Image"**
3. **Select an image** from `prepared_dataset/images/` (e.g., `char_000001.png`)
4. **Click "Recognize Text"**
5. **Expected result**:
   - Should show recognized Ranjana character
   - Should display confidence score
   - Should show bounding box (if multiple characters)

### Test 2: Capture from Camera

1. **Open frontend**: http://localhost:5173
2. **Click "Camera" tab**
3. **Allow camera permissions**
4. **Point camera at Ranjana script text**
5. **Click "Capture"**
6. **Click "Recognize Text"**
7. **Expected result**: Recognized text appears

### Test 3: Test with Real Ranjana Script Image

1. **Take a photo** of Ranjana script text
2. **Upload** via frontend
3. **Check recognition accuracy**

---

## Troubleshooting

### Issue 1: "OCR Service is not responding"

**Symptoms:**
- Frontend shows error: "OCR Service is not responding..."
- Error in browser console

**Solutions:**

1. **Check Python OCR service is running:**
   ```bash
   curl http://localhost:5000/health
   ```
   Should return: `{"status":"healthy","model_loaded":true,...}`

2. **Check Java backend is running:**
   ```bash
   curl http://localhost:8080/api/health
   ```
   Should return: `{"success":true,...}`

3. **Check Java backend can reach Python service:**
   - Look at Java backend logs
   - Should see successful connection to Python service

4. **Check CORS configuration:**
   - Python service should allow CORS (already configured)
   - Java backend should allow CORS (already configured)

---

### Issue 2: "Error processing image"

**Symptoms:**
- Frontend shows "Error processing image"
- Console shows error details

**Solutions:**

1. **Check image format:**
   - Supported: PNG, JPG, JPEG
   - Try a different image

2. **Check Python OCR service logs:**
   - Look for error messages
   - Common issues:
     - Model not loaded
     - Image format not supported
     - Image too large

3. **Check Java backend logs:**
   - Look for error messages
   - Common issues:
     - Cannot connect to Python service
     - Image size exceeded

---

### Issue 3: Wrong Predictions

**Symptoms:**
- OCR returns wrong characters
- Low confidence scores

**Solutions:**

1. **Image quality:**
   - Use clear, high-contrast images
   - Ensure text is readable
   - Avoid blurry images

2. **Single character images:**
   - Works best with single characters
   - Multi-character images may need better segmentation

3. **Model accuracy:**
   - Current accuracy: 96% on training data
   - Real-world images may have lower accuracy
   - Consider retraining with more diverse data

---

### Issue 4: Frontend Not Loading

**Symptoms:**
- Browser shows blank page
- Console errors

**Solutions:**

1. **Check frontend is running:**
   ```bash
   cd frontend
   npm run dev
   ```

2. **Check dependencies:**
   ```bash
   cd frontend
   npm install
   ```

3. **Clear browser cache:**
   - Hard refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)

4. **Check console for errors:**
   - Open browser DevTools (F12)
   - Check Console tab for errors

---

## Verification Checklist

Before testing, verify:

- [ ] Python OCR service is running (port 5000)
- [ ] Model is loaded successfully
- [ ] Java backend is running (port 8080)
- [ ] Java backend can connect to Python service
- [ ] React frontend is running (port 5173)
- [ ] Browser can access http://localhost:5173
- [ ] Camera permissions granted (if testing camera)

---

## Expected Results

### Good Test Results:
- ✅ Recognized text appears quickly (< 2 seconds)
- ✅ Confidence scores > 0.8
- ✅ Correct Ranjana characters displayed
- ✅ Bounding boxes shown (for AR visualization)

### Poor Test Results:
- ⚠️ Recognition takes > 5 seconds
- ⚠️ Confidence scores < 0.5
- ⚠️ Wrong characters predicted
- ⚠️ "Error processing image" messages

---

## Quick Test Commands

### Test Python OCR Service Directly:

```bash
# Test health
curl http://localhost:5000/health

# Test OCR (Windows PowerShell)
$imageBytes = [System.IO.File]::ReadAllBytes("prepared_dataset/images/char_000001.png")
$base64 = [System.Convert]::ToBase64String($imageBytes)
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "multipart/form-data" -Form @{image=Get-Item "prepared_dataset/images/char_000001.png"}
```

### Test Java Backend:

```bash
# Test health
curl http://localhost:8080/api/health

# Test OCR endpoint (with image file)
curl -X POST http://localhost:8080/api/ocr/recognize \
  -F "image=@prepared_dataset/images/char_000001.png"
```

---

## Next Steps After Testing

1. **If everything works:**
   - ✅ You're ready to use the OCR system!
   - ✅ Test with real-world images
   - ✅ Collect feedback for improvements

2. **If issues occur:**
   - Check logs from all 3 services
   - Verify network connectivity
   - Check file permissions
   - Review error messages

---

## Performance Tips

1. **Image Size:**
   - Keep images < 5MB
   - Resize very large images before upload

2. **Network:**
   - All services on same machine = fastest
   - Network latency affects response time

3. **Model:**
   - First prediction takes longer (model loading)
   - Subsequent predictions are faster

---

**Ready to test! Start all 3 services and open the frontend!** 🚀
