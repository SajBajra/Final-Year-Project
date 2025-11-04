# 🚀 Quick Guide: Start Java Backend

## Problem
The frontend shows: "OCR Service is not responding" because the **Java Backend is not running**.

## Solution: Start Java Backend

### Option 1: Command Line (Maven)

```bash
cd javabackend
mvn spring-boot:run
```

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

Wait until you see: `Started LipikaApplication` (this takes 10-30 seconds)

---

### Option 2: Eclipse IDE

1. **Open Eclipse**
2. **Import project** (if not already imported):
   - File → Open Projects from File System
   - Select `javabackend` folder
3. **Run the application**:
   - Right-click on `src/main/java/com/lipika/LipikaApplication.java`
   - Run As → Java Application
4. **Check Console** for startup messages

---

### Option 3: Use Startup Script

Run the script that starts all services:
```powershell
.\START_ALL_SERVICES.ps1
```

---

## Verify Backend is Running

After starting, check if it's working:

### Method 1: Browser
Open: http://localhost:8080/api/health

Should see:
```json
{
  "success": true,
  "message": "All services are healthy",
  "data": null
}
```

### Method 2: PowerShell
```powershell
Invoke-WebRequest -Uri "http://localhost:8080/api/health"
```

---

## Architecture Reminder

The frontend requires **BOTH** services:

```
Frontend (React)
    ↓
Java Backend (port 8080) ← YOU NEED TO START THIS
    ↓
Python OCR Service (port 5000) ← Already running ✓
```

---

## Troubleshooting

### Port 8080 Already in Use

If you get: `Port 8080 is already in use`

**Solution 1:** Stop the existing process
```powershell
# Find process using port 8080
netstat -ano | findstr :8080

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

**Solution 2:** Change port in `javabackend/src/main/resources/application.properties`:
```properties
server.port=8081
```

Then update frontend `src/services/ocrService.js`:
```javascript
const API_BASE_URL = 'http://localhost:8081/api'
```

---

### Maven Not Found

If `mvn` command not found:

**Solution:** Use Eclipse instead (Option 2 above)

Or install Maven:
1. Download from: https://maven.apache.org/download.cgi
2. Add to PATH

---

### Lombok Errors in Eclipse

If you see Lombok-related compilation errors:

1. Install Lombok plugin in Eclipse:
   - Help → Eclipse Marketplace
   - Search "Lombok"
   - Install
2. Enable annotation processing:
   - Project → Properties → Java Compiler → Annotation Processing
   - Check "Enable annotation processing"

See `javabackend/LOMBOK_SETUP.md` for detailed instructions.

---

## Once Backend is Running

1. ✅ Java Backend running on port 8080
2. ✅ Python OCR Service running on port 5000
3. ✅ Frontend running on port 5173

**Now try uploading an image in the frontend - it should work!** 🎉
