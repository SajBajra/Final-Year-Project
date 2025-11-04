# 📦 Node.js Installation Guide for Lipika Frontend

## 🎯 Purpose

The React frontend requires Node.js to run. This guide will help you install Node.js and get the frontend running.

---

## 📥 Step 1: Install Node.js

### Option A: Download and Install (Recommended)

1. **Visit the Node.js website:**
   - Go to: https://nodejs.org/
   - Download the **LTS version** (Long Term Support)
   - Current LTS: v20.x or v18.x

2. **Run the installer:**
   - Double-click the downloaded `.msi` file
   - Follow the installation wizard
   - **Important:** Check "Add to PATH" during installation (usually checked by default)
   - Click "Next" through all steps
   - Click "Install" (may require administrator privileges)

3. **Verify installation:**
   - Close and reopen your terminal/PowerShell
   - Run: `node --version`
   - Run: `npm --version`
   - You should see version numbers (e.g., `v20.11.0` and `10.2.4`)

### Option B: Using Chocolatey (If you have it)

```powershell
choco install nodejs-lts
```

### Option C: Using winget (Windows Package Manager)

```powershell
winget install OpenJS.NodeJS.LTS
```

---

## ✅ Step 2: Verify Installation

After installing Node.js, **close and reopen your terminal**, then run:

```powershell
node --version
npm --version
```

**Expected Output:**
```
v20.11.0
10.2.4
```

If you see version numbers, Node.js is installed correctly!

---

## 🚀 Step 3: Install Frontend Dependencies

Once Node.js is installed, navigate to the frontend directory and install dependencies:

```powershell
cd frontend
npm install
```

**Expected Output:**
```
added 150+ packages, and audited 150+ packages in 30s
found 0 vulnerabilities
```

⏱️ **Time:** 30-60 seconds

---

## ▶️ Step 4: Start the Frontend

After dependencies are installed, start the development server:

```powershell
npm run dev
```

**Expected Output:**
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

✅ **Frontend URL:** http://localhost:5173

---

## 🎉 Step 5: Access the Frontend

1. Open your browser
2. Navigate to: **http://localhost:5173**
3. You should see the Lipika homepage with:
   - Upload image option
   - Camera capture option
   - AR overlay features

---

## 🔧 Troubleshooting

### Issue: "node is not recognized"

**Solution:**
- Close and reopen your terminal/PowerShell
- If still not working, add Node.js to PATH manually:
  1. Search "Environment Variables" in Windows
  2. Edit "Path" variable
  3. Add: `C:\Program Files\nodejs\`
  4. Restart terminal

### Issue: "npm install" fails

**Possible Causes:**
- Network issues
- Permission issues
- Corrupted npm cache

**Solutions:**
```powershell
# Clear npm cache
npm cache clean --force

# Try installing again
npm install
```

### Issue: Port 5173 already in use

**Solution:**
```powershell
# Find process using port 5173
netstat -ano | findstr ":5173"

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or change port in vite.config.js
```

---

## 📋 Quick Reference

### Complete Setup Commands

```powershell
# 1. Verify Node.js is installed
node --version
npm --version

# 2. Navigate to frontend
cd frontend

# 3. Install dependencies (first time only)
npm install

# 4. Start frontend
npm run dev

# 5. Open browser
# Go to: http://localhost:5173
```

---

## 🎯 What You'll Get

After completing these steps:

✅ **Frontend running on:** http://localhost:5173  
✅ **Upload images** for OCR  
✅ **Camera capture** for real-time OCR  
✅ **AR overlay** with bounding boxes  
✅ **Beautiful UI** with Tailwind CSS  

---

## 🔗 Integration with OCR Service

The frontend automatically connects to your OCR service at:
- **OCR Service:** http://localhost:5000
- **Frontend:** http://localhost:5173

**Make sure the OCR service is running** before using the frontend!

To start OCR service:
```powershell
cd python-model
python ocr_service_ar.py
```

---

## 📝 Next Steps After Installation

1. ✅ Install Node.js (this guide)
2. ✅ Install frontend dependencies (`npm install`)
3. ✅ Start frontend (`npm run dev`)
4. ✅ Verify OCR service is running (http://localhost:5000/health)
5. ✅ Test full integration (upload Ranjana image)

---

**Need help?** Check the main README.md or SERVICES_RUNNING.md files.

---

**Last Updated:** Installation guide creation
**Node.js Version:** LTS 20.x or 18.x recommended
