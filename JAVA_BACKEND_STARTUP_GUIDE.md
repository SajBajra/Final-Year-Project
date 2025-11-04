# 🚀 Java Backend Startup Guide - Complete Troubleshooting

## Current Status
- ✅ **Python OCR Service**: Running on port 5000 (Model loaded: 25 characters)
- ❌ **Java Backend**: NOT running on port 8080

---

## ⚠️ Common Issues & Solutions

### Issue 1: Java Backend Won't Start

**Symptoms:**
- Port 8080 is not responding
- Timeout errors when trying to access `/api/health`
- Frontend shows "OCR Service is not responding"

**Solution: Start the Java Backend**

#### **Method A: Using Eclipse (Recommended)**

1. **Open Eclipse IDE**

2. **Ensure Project is Imported:**
   - File → Import → Existing Maven Projects
   - Browse to `E:\Cllz\FYP\javabackend`
   - Click Finish

3. **Update Maven Dependencies:**
   - Right-click `javabackend` project
   - Maven → Update Project...
   - Check **Force Update of Snapshots/Releases**
   - Click OK
   - **Wait for dependencies to download** (1-2 minutes)

4. **Clean and Build:**
   - Project → Clean...
   - Select `javabackend`
   - Click Clean

5. **Start the Application:**
   - Navigate to: `src/main/java/com/lipika/LipikaApplication.java`
   - Right-click `LipikaApplication.java`
   - **Run As** → **Spring Boot App**
   - **OR** **Run As** → **Java Application**

6. **Check Console Output:**
   ```
   .   ____          _            __ _ _
    /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
   ( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
    \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
     '  |____| .__|_| |_|_| |_\__, | / / / /
    =========|_|==============|___/=/_/_/_/
    :: Spring Boot ::                (v3.x.x)
   
   2025-01-XX - Starting LipikaApplication
   ...
   2025-01-XX - Started LipikaApplication in X.XXX seconds
   Tomcat started on port(s): 8080 (http)
   ```

7. **Verify It's Running:**
   - Open browser: http://localhost:8080/api/health
   - Should return: `{"success":true,"message":"OCR service is healthy"}`

---

#### **Method B: Using Command Line (Alternative)**

```powershell
cd E:\Cllz\FYP\javabackend
mvn clean install
mvn spring-boot:run
```

---

### Issue 2: Compilation Errors

**Error:** `getSuccess() is undefined`

**Solution:** Already fixed! Use `isSuccess()` instead of `getSuccess()` for boolean fields.

**If you still see errors:**
1. Project → Clean → Clean all projects
2. Right-click project → Maven → Update Project → Force Update
3. Restart Eclipse

---

### Issue 3: Port 8080 Already in Use

**Error:** `Port 8080 is already in use`

**Solution:**
```powershell
# Find what's using port 8080
netstat -ano | findstr ":8080"

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

**OR** Change port in `application.properties`:
```properties
server.port=8081
```

---

### Issue 4: Lombok Not Working

**Error:** `log cannot be resolved` or `@Slf4j not working`

**Solution:**
1. Download Lombok: https://projectlombok.org/download
2. Run: `java -jar lombok.jar`
3. Select Eclipse installation directory
4. Click Install/Update
5. Restart Eclipse
6. Project → Clean → Clean all projects

---

### Issue 5: Maven Dependencies Not Downloading

**Solution:**
1. Window → Preferences → Maven → User Settings
2. Check User settings path
3. Click Update Settings
4. Right-click project → Maven → Update Project → Force Update

---

## ✅ Verification Steps

After starting the Java backend:

### Step 1: Check Java Backend Health
```powershell
Invoke-WebRequest -Uri http://localhost:8080/api/health -UseBasicParsing
```

**Expected Response:**
```json
{
  "success": true,
  "message": "OCR service is healthy",
  "data": "OCR service is healthy"
}
```

### Step 2: Check Python OCR Service
```powershell
Invoke-WebRequest -Uri http://localhost:5000/health -UseBasicParsing
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "chars_count": 25
}
```

### Step 3: Test Full Flow
1. Open frontend: http://localhost:5173
2. Upload a Ranjana image
3. Should see OCR results

---

## 🔧 Quick Diagnostic Script

Run this PowerShell script to check everything:

```powershell
Write-Host "=== Lipika Service Diagnostics ===" -ForegroundColor Cyan

# Check Python OCR Service
Write-Host "`n1. Python OCR Service (Port 5000):" -ForegroundColor Yellow
try {
    $py = Invoke-WebRequest -Uri http://localhost:5000/health -UseBasicParsing -TimeoutSec 3
    $health = $py.Content | ConvertFrom-Json
    Write-Host "   [OK] Running" -ForegroundColor Green
    Write-Host "   Model Loaded: $($health.model_loaded)" -ForegroundColor Gray
    Write-Host "   Characters: $($health.chars_count)" -ForegroundColor Gray
} catch {
    Write-Host "   [ERROR] Not responding" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
}

# Check Java Backend
Write-Host "`n2. Java Backend (Port 8080):" -ForegroundColor Yellow
try {
    $java = Invoke-WebRequest -Uri http://localhost:8080/api/health -UseBasicParsing -TimeoutSec 3
    Write-Host "   [OK] Running" -ForegroundColor Green
    Write-Host "   Response: $($java.Content)" -ForegroundColor Gray
} catch {
    Write-Host "   [ERROR] Not responding" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host "`n   ACTION REQUIRED:" -ForegroundColor Yellow
    Write-Host "   1. Open Eclipse" -ForegroundColor White
    Write-Host "   2. Right-click LipikaApplication.java" -ForegroundColor White
    Write-Host "   3. Run As → Spring Boot App" -ForegroundColor White
}

# Check Ports
Write-Host "`n3. Port Status:" -ForegroundColor Yellow
$port5000 = netstat -ano | findstr ":5000" | Select-Object -First 1
$port8080 = netstat -ano | findstr ":8080" | Select-Object -First 1
if ($port5000) {
    Write-Host "   Port 5000: [IN USE]" -ForegroundColor Green
} else {
    Write-Host "   Port 5000: [FREE]" -ForegroundColor Red
}
if ($port8080) {
    Write-Host "   Port 8080: [IN USE]" -ForegroundColor Green
} else {
    Write-Host "   Port 8080: [FREE]" -ForegroundColor Red
    Write-Host "   → Java backend needs to be started!" -ForegroundColor Yellow
}

Write-Host "`n=== End Diagnostics ===" -ForegroundColor Cyan
```

---

## 📝 Step-by-Step Startup Checklist

- [ ] Eclipse is open
- [ ] `javabackend` project is imported
- [ ] Maven dependencies are downloaded (check bottom-right corner)
- [ ] Project builds without errors (check Problems tab)
- [ ] `LipikaApplication.java` exists
- [ ] Right-clicked `LipikaApplication.java` → Run As → Spring Boot App
- [ ] Console shows "Started LipikaApplication"
- [ ] Console shows "Tomcat started on port(s): 8080"
- [ ] Browser test: http://localhost:8080/api/health returns success

---

## 🎯 If Still Not Working

1. **Check Eclipse Console for Errors:**
   - Look for red error messages
   - Copy and share the error if you need help

2. **Verify Java Version:**
   - Window → Preferences → Java → Installed JREs
   - Should have Java 17 or 21

3. **Check Project Properties:**
   - Right-click project → Properties
   - Java Build Path → Libraries
   - Should show Maven Dependencies

4. **Try Fresh Start:**
   - Close Eclipse
   - Delete `.classpath` and `.project` files (if they exist)
   - Re-import project in Eclipse
   - Clean and build again

---

## 💡 Pro Tip: Create Run Configuration

1. Right-click `LipikaApplication.java`
2. Run As → Run Configurations...
3. Create new **Java Application**:
   - Name: `Lipika Backend`
   - Main class: `com.lipika.LipikaApplication`
   - Project: `javabackend`
4. Click **Run**

Now you can easily start it from the Run button dropdown!

---

**Need More Help?** Check the console output in Eclipse and share any error messages you see.
