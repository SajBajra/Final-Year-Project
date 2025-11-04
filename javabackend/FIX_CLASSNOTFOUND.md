# 🔧 Fix: Could not find or load main class

## Problem
```
Error: Could not find or load main class com.lipika.LipikaApplication
Caused by: java.lang.ClassNotFoundException: com.lipika.LipikaApplication
```

This means the Java classes haven't been compiled yet!

---

## ✅ Solution: Build the Project First

### Option 1: Use Maven Spring Boot Plugin (Recommended)

This automatically compiles and runs:

```bash
cd javabackend
mvn spring-boot:run
```

This command will:
1. ✅ Compile the project automatically
2. ✅ Download dependencies (first time only)
3. ✅ Run the application

**Wait for it to finish building** - first run may take 2-5 minutes to download dependencies.

---

### Option 2: Build First, Then Run

If `mvn spring-boot:run` doesn't work, build manually first:

```bash
cd javabackend

# Step 1: Clean previous builds
mvn clean

# Step 2: Compile and package
mvn package

# Step 3: Run Spring Boot
mvn spring-boot:run
```

Or run the JAR directly:
```bash
java -jar target/lipika-0.0.1-SNAPSHOT.jar
```

---

### Option 3: Build in Eclipse

1. **Right-click on `javabackend` project** in Eclipse
2. Select **Maven → Update Project**
   - Check "Force Update of Snapshots/Releases"
   - Click OK
3. **Right-click on `javabackend` project** again
4. Select **Run As → Maven build...**
5. Enter goals: `clean package`
6. Click Run
7. Wait for "BUILD SUCCESS"
8. **Then** right-click `LipikaApplication.java` → Run As → Java Application

---

## Step-by-Step Troubleshooting

### Step 1: Verify Project Structure

Make sure you have this structure:
```
javabackend/
├── pom.xml
├── src/
│   └── main/
│       └── java/
│           └── com/
│               └── lipika/
│                   └── LipikaApplication.java
```

### Step 2: Check Java Version

```bash
java -version
```

Should show Java 17 or higher (required for Spring Boot 3.x)

If wrong version, update `pom.xml` or install Java 17.

### Step 3: Verify Maven is Working

```bash
cd javabackend
mvn -version
```

Should show Maven version.

### Step 4: Clean Build

```bash
cd javabackend
mvn clean install
```

Wait for "BUILD SUCCESS"

### Step 5: Verify Classes Were Compiled

```bash
# Windows PowerShell
Test-Path "javabackend/target/classes/com/lipika/LipikaApplication.class"

# Should return: True
```

### Step 6: Run Application

```bash
mvn spring-boot:run
```

---

## Common Issues

### Issue 1: "mvn command not found"

**Solution:** Install Maven or use Eclipse:
- Install Maven: https://maven.apache.org/download.cgi
- Or use Eclipse (it has built-in Maven support)

---

### Issue 2: "Java version mismatch"

**Error:** `Unsupported class file major version`

**Solution:** 
- Check `pom.xml` has: `<java.version>17</java.version>`
- Install Java 17+ from: https://adoptium.net/

---

### Issue 3: "Dependencies download failing"

**Error:** Failed to download dependencies

**Solution:**
```bash
# Clean Maven cache
mvn clean
rm -rf ~/.m2/repository  # Linux/Mac
# Or delete C:\Users\YourName\.m2\repository on Windows

# Rebuild
mvn clean install
```

---

### Issue 4: "Port 8080 already in use"

**Error:** `Web server failed to start. Port 8080 was already in use`

**Solution:**
```powershell
# Find process using port 8080
netstat -ano | findstr :8080

# Kill the process (replace PID)
taskkill /PID <PID> /F
```

---

## Quick Fix Script

Run this PowerShell script in the project root:

```powershell
Write-Host "Building Java Backend..." -ForegroundColor Yellow
cd javabackend

Write-Host "Step 1: Cleaning..." -ForegroundColor Cyan
mvn clean

Write-Host "Step 2: Compiling..." -ForegroundColor Cyan
mvn compile

Write-Host "Step 3: Packaging..." -ForegroundColor Cyan
mvn package -DskipTests

if (Test-Path "target/classes/com/lipika/LipikaApplication.class") {
    Write-Host "[SUCCESS] Build complete! Classes found." -ForegroundColor Green
    Write-Host "Now run: mvn spring-boot:run" -ForegroundColor Yellow
} else {
    Write-Host "[ERROR] Build failed. Classes not found." -ForegroundColor Red
    Write-Host "Check the error messages above." -ForegroundColor Yellow
}

cd ..
```

---

## Verify It's Working

After building and running, check:

1. **Browser:** http://localhost:8080/api/health
   - Should return JSON response

2. **PowerShell:**
   ```powershell
   Invoke-WebRequest -Uri "http://localhost:8080/api/health"
   ```

3. **Look for in console:**
   ```
   Started LipikaApplication in X.XXX seconds
   ```

---

## Still Not Working?

1. **Check the full error message** - look for specific dependency issues
2. **Try Eclipse** - it handles Maven builds automatically
3. **Check Java version** - must be Java 17+
4. **Check pom.xml** - verify it's valid XML
5. **Check if `target/` folder exists** - if not, Maven hasn't built yet

---

**Most Common Fix:** Just run `mvn spring-boot:run` from the `javabackend` folder - it will compile automatically! 🚀
