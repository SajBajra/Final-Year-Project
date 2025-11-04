# 🚀 Running Services Guide - Complete Setup

## 📋 Service Architecture

Your Lipika system has **3 layers** that all need to be running:

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   React     │         │  Java Backend │         │   Python    │
│  Frontend   │────────▶│  (Spring Boot)│────────▶│  OCR Service│
│             │         │               │         │             │
│ Port 3000/  │◀────────│  Port 8080    │◀────────│  Port 5000  │
│   5173      │         │               │         │             │
└─────────────┘         └──────────────┘         └─────────────┘
     │                        │                         │
     └────────────────────────┴─────────────────────────┘
                    ALL MUST BE RUNNING!
```

### Why You Need All 3 Services:

1. **Frontend (React)** - User interface
2. **Java Backend (Spring Boot)** - Middleware that:
   - Receives requests from frontend
   - Validates and processes data
   - Calls Python OCR service
   - Returns formatted responses
3. **Python OCR Service** - Actually performs OCR recognition

---

## ✅ Answer to Your Questions

### Q1: Why "Error Processing Image" When Only Frontend Runs?

**Answer: YES, it's because the Java backend isn't running!**

The frontend is configured to call:
```javascript
API_BASE_URL = 'http://localhost:8080/api'
```

So when you upload an image:
1. Frontend tries to call `http://localhost:8080/api/ocr/recognize`
2. If Java backend isn't running → Connection refused
3. Error: "OCR Service is not responding. Please check if the Java backend is running on port 8080."

**Solution:** You **MUST** start the Java backend for the frontend to work!

---

### Q2: Do I Need XAMPP Tomcat?

**Answer: NO, you don't need XAMPP's Tomcat!**

**Why:**
- Spring Boot includes an **embedded Tomcat server**
- When you run Spring Boot from Eclipse, it starts its own Tomcat on port 8080
- XAMPP's Tomcat is a separate Apache Tomcat installation (not needed)

**What XAMPP Provides:**
- **Apache** - Web server (not needed for Spring Boot)
- **MySQL** - Database (only needed if you're using a database)
- **Tomcat** - Java servlet container (not needed, Spring Boot has its own)
- **phpMyAdmin** - MySQL admin tool (only needed if using MySQL)

**For Your Project:**
- ✅ **You need**: Java backend running (Spring Boot with embedded Tomcat)
- ❌ **You don't need**: XAMPP Tomcat
- ❌ **You don't need**: XAMPP Apache
- ❓ **MySQL**: Only if you add database features later

---

## 🚀 How to Run All Services

### Step 1: Start Python OCR Service

**Terminal 1:**
```bash
cd python-model
python ocr_service_ar.py
```

**Expected Output:**
```
============================================================
AR-Ready Ranjana Script OCR Service
============================================================
Loading ImprovedCharacterCRNN model from best_character_crnn.pth...
✓ Model loaded successfully! Type: ImprovedCharacterCRNN, Characters: XX
Device: cpu
Service running on http://0.0.0.0:5000
============================================================
```

✅ **Verify:** Open http://localhost:5000/health in browser

---

### Step 2: Start Java Backend (Spring Boot)

**Option A: Using Eclipse (Recommended for You)**

1. Open Eclipse
2. Import/Open the `javabackend` project
3. Find `LipikaApplication.java` (main class)
4. Right-click → **Run As** → **Java Application**
   OR
   Right-click → **Run As** → **Spring Boot App**

**Expected Output:**
```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::                (v2.x.x)

2025-01-XX XX:XX:XX - Starting LipikaApplication
2025-01-XX XX:XX:XX - Started LipikaApplication in X.XXX seconds
Tomcat started on port(s): 8080 (http)
```

**Option B: Using Maven (Command Line)**

```bash
cd javabackend
mvn spring-boot:run
```

✅ **Verify:** Open http://localhost:8080/api/health in browser

---

### Step 3: Start Frontend

**Terminal 2 (or Terminal 3 if Java is in Eclipse):**
```bash
cd frontend
npm run dev
```

**Expected Output:**
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:5173/
```

✅ **Verify:** Open http://localhost:5173 in browser

---

## ✅ Verification Checklist

Before testing, verify all services are running:

1. ✅ **Python OCR Service** - http://localhost:5000/health
   ```json
   {
     "status": "healthy",
     "model_loaded": true,
     "chars_count": XX
   }
   ```

2. ✅ **Java Backend** - http://localhost:8080/api/health
   ```json
   {
     "success": true,
     "data": {
       "status": "UP",
       "service": "Lipika Backend - Presenter Layer"
     }
   }
   ```

3. ✅ **Frontend** - http://localhost:5173
   - Should see Lipika homepage
   - Upload an image
   - Should work without "error processing image"

---

## 🔄 Request Flow

When you upload an image:

```
1. Frontend (Port 5173)
   ↓ POST /api/ocr/recognize
   ↓ Image file
   
2. Java Backend (Port 8080)
   ↓ Receives request
   ↓ Validates file
   ↓ POST /predict
   ↓ Image file
   
3. Python OCR Service (Port 5000)
   ↓ Processes image
   ↓ Runs OCR model
   ↓ Returns OCR results
   
4. Java Backend (Port 8080)
   ↓ Formats response
   ↓ Returns JSON
   
5. Frontend (Port 5173)
   ↓ Displays results
   ↓ Shows AR overlay
```

**If any service is missing, the chain breaks!**

---

## 🐛 Common Errors & Solutions

### Error: "Error Processing Image" / "OCR Service is not responding"

**Cause:** Java backend not running

**Solution:**
1. Check if Java backend is running: http://localhost:8080/api/health
2. If not running, start it from Eclipse or with `mvn spring-boot:run`
3. Make sure it's on port 8080 (check console output)

---

### Error: "Connection refused" on port 8080

**Cause:** Java backend not started or wrong port

**Solution:**
- Start Java backend
- Check `javabackend/src/main/resources/application.properties`:
  ```properties
  server.port=8080
  ```

---

### Error: "Connection refused" on port 5000

**Cause:** Python OCR service not running

**Solution:**
```bash
cd python-model
python ocr_service_ar.py
```

---

### Error: "Model not loaded" in Python service

**Cause:** Model file missing or wrong path

**Solution:**
- Make sure `best_character_crnn.pth` exists in `python-model/` folder
- Check console output for model loading messages

---

## 📝 Quick Start Command Summary

**Terminal 1 - Python OCR:**
```bash
cd python-model && python ocr_service_ar.py
```

**Eclipse - Java Backend:**
- Right-click `LipikaApplication.java` → Run As → Spring Boot App

**Terminal 2 - Frontend:**
```bash
cd frontend && npm run dev
```

---

## 🎯 Summary

| Service | Port | Required? | How to Start |
|---------|------|-----------|--------------|
| Python OCR | 5000 | ✅ YES | `python ocr_service_ar.py` |
| Java Backend | 8080 | ✅ YES | Eclipse or `mvn spring-boot:run` |
| Frontend | 5173 | ✅ YES | `npm run dev` |
| XAMPP Tomcat | 8080 | ❌ NO | Not needed (Spring Boot has embedded Tomcat) |
| XAMPP Apache | 80 | ❌ NO | Not needed |
| MySQL | 3306 | ❌ NO | Only if using database (future) |

**All 3 services (Python, Java, Frontend) must be running for the app to work!**

---

## 💡 Pro Tip: Create a Startup Script

You can create a script to start all services automatically:

**`start_all.ps1`** (Windows PowerShell):
```powershell
# Start Python OCR Service
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd python-model; python ocr_service_ar.py"

# Start Frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Write-Host "Started Python OCR and Frontend services"
Write-Host "Now start Java Backend from Eclipse!"
```

Then manually start Java backend from Eclipse.

---

**Remember: If you see "Error Processing Image", check that ALL 3 services are running!** ✅
