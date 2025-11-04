# 🧪 Lipika Complete System Testing Guide

## ✅ Prerequisites

- ✅ Model trained: `best_character_crnn.pth` exists in `python-model/`
- ✅ Dependencies installed: `pip install -r requirements.txt` (Python)
- ✅ Frontend dependencies: `npm install` (in frontend folder)

---

## 🚀 Testing Steps

### Step 1: Start OCR Service (Terminal 1)

```bash
cd python-model
python ocr_service_ar.py
```

**Expected Output:**
```
Loading model from best_character_crnn.pth...
Model loaded successfully!
Character set: 82 characters
 * Serving Flask app 'ocr_service_ar'
 * Running on http://127.0.0.1:5000
```

✅ **Service Running**: http://localhost:5000

---

### Step 2: Start Frontend (Terminal 2)

Open a **NEW** terminal window:

```bash
cd frontend
npm run dev
```

**Expected Output:**
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

✅ **Frontend Running**: http://localhost:3000

---

## 🧪 Test Scenarios

### Test 1: Basic Health Check

1. **Check OCR Service Health:**
   ```
   Open browser: http://localhost:5000/health
   ```
   Should return: `{"status": "ok"}`

2. **Check Frontend:**
   ```
   Open browser: http://localhost:3000
   ```
   Should see: Lipika homepage with upload and camera options

---

### Test 2: Image Upload & OCR

1. Go to http://localhost:3000
2. Click on **"Upload Image"** card
3. Drag & drop or select a Ranjana script image
4. Wait for processing...
5. **Expected Results:**
   - ✅ Recognized text displayed
   - ✅ Character breakdown shown
   - ✅ Confidence score displayed
   - ✅ "Show AR Overlay" button appears

---

### Test 3: AR Overlay Visualization

1. After successful OCR, click **"👓 Show AR Overlay"**
2. **Expected Results:**
   - ✅ Image displayed with bounding boxes
   - ✅ Blue boxes around each character
   - ✅ Hover over boxes shows character labels
   - ✅ Confidence scores in tooltips

---

### Test 4: Camera Capture

1. Click on **"Camera Capture"** card
2. Click **"Start Camera"** button
3. Grant camera permissions if prompted
4. Click **"📸 Capture"** button
5. **Expected Results:**
   - ✅ Captured image displayed
   - ✅ OCR processing starts automatically
   - ✅ Results displayed as in Test 2

---

## 🔍 Verification Checklist

### OCR Service Verification

- [ ] Service starts without errors
- [ ] Model loads successfully (check logs)
- [ ] Health endpoint responds: `GET /health`
- [ ] Predict endpoint accepts images: `POST /predict`
- [ ] Returns JSON with `text`, `characters`, `confidence`

### Frontend Verification

- [ ] Page loads without errors
- [ ] All components render correctly
- [ ] Image upload works
- [ ] Camera capture works (if permissions granted)
- [ ] OCR results display correctly
- [ ] AR overlay displays bounding boxes
- [ ] Responsive design works on different screen sizes

### Integration Verification

- [ ] Frontend can communicate with OCR service
- [ ] Images are sent correctly to backend
- [ ] Results are displayed in real-time
- [ ] Error handling works (try invalid image)
- [ ] Loading states appear during processing

---

## 🐛 Troubleshooting

### Issue: OCR Service Won't Start

**Error:** `Model file not found`

**Solution:**
```bash
# Check if model exists
ls python-model/best_character_crnn.pth

# If missing, train the model first
cd python-model
python train_character_crnn.py --epochs 100
```

---

### Issue: Frontend Won't Start

**Error:** `npm: command not found`

**Solution:**
- Install Node.js from https://nodejs.org/
- Then: `npm install` in frontend folder

**Error:** `Module not found`

**Solution:**
```bash
cd frontend
npm install
```

---

### Issue: CORS Errors

**Error:** `Access-Control-Allow-Origin` error

**Solution:**
- Check `ocr_service_ar.py` has `CORS(app)`
- Verify Flask-CORS is installed: `pip install flask-cors`

---

### Issue: Images Not Processing

**Error:** No response from OCR service

**Check:**
1. OCR service is running: http://localhost:5000/health
2. Check browser console for errors (F12)
3. Verify image format (JPG, PNG, etc.)
4. Check OCR service logs for errors

---

### Issue: AR Overlay Not Showing

**Check:**
1. OCR returned character data with `bbox` fields
2. Check browser console for JavaScript errors
3. Verify `characters` array contains bounding box data
4. Try with different image

---

## 📊 Expected Performance

- **OCR Processing**: < 2 seconds per image
- **Model Loading**: ~1-2 seconds on start
- **Frontend Response**: < 100ms for UI updates
- **AR Rendering**: Real-time, no lag

---

## 🎯 Success Criteria

Your system is working correctly if:

1. ✅ Both services start without errors
2. ✅ Can upload and process images
3. ✅ OCR results are accurate for Ranjana text
4. ✅ AR overlay shows bounding boxes correctly
5. ✅ All UI components are functional
6. ✅ No console errors in browser
7. ✅ Responsive design works

---

## 📝 Test Report Template

```
Date: ___________
Tester: ___________

✅ Services Started:
- OCR Service: [ ] Running on :5000
- Frontend: [ ] Running on :3000

✅ Functionality:
- Image Upload: [ ] Pass / [ ] Fail
- OCR Recognition: [ ] Pass / [ ] Fail
- AR Overlay: [ ] Pass / [ ] Fail
- Camera Capture: [ ] Pass / [ ] Fail

✅ Performance:
- Processing Time: _____ seconds
- UI Responsiveness: [ ] Good / [ ] Needs Improvement

Issues Found:
1. _________________________________
2. _________________________________

Notes:
___________________________________
```

---

## 🎉 Next Steps After Successful Testing

1. **Optimize Model** (if needed)
   - Fine-tune for specific use cases
   - Add more training data

2. **Deploy to Production**
   - Deploy OCR service (Flask → Gunicorn)
   - Deploy frontend (Vite build → Nginx/Netlify)
   - Set up CI/CD pipeline

3. **Add Features**
   - Translation support
   - Text-to-speech
   - Export functionality
   - User authentication

4. **Build Java Backend** (MVP Presenter Layer)
   - Spring Boot REST API
   - Business logic layer
   - Database integration

---

**🎯 Happy Testing! Your Lipika system should be fully functional now!**
