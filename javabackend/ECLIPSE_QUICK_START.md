# 🚀 Eclipse Quick Start Guide

## Problem
- Error: "Could not find or load main class com.lipika.LipikaApplication"
- Maven command not found

## ✅ Solution: Use Eclipse (No Maven Command Line Needed!)

Eclipse has built-in Maven support, so you don't need to install Maven separately.

---

## Step-by-Step Instructions

### Step 1: Open Eclipse

1. Launch Eclipse IDE
2. If prompted, select a workspace (or use default)

---

### Step 2: Import the Project

**Method A: Import Existing Maven Project**
1. File → Import
2. Select: **Maven → Existing Maven Projects**
3. Click Next
4. Browse to: `E:\Cllz\FYP\javabackend` (select the folder containing `pom.xml`)
5. Click Finish

**Method B: Open Projects from File System**
1. File → Open Projects from File System...
2. Click Directory...
3. Browse to: `E:\Cllz\FYP\javabackend`
4. Click Finish

---

### Step 3: Wait for Maven to Build

Eclipse will automatically:
- ✅ Download Maven dependencies (may take 2-5 minutes first time)
- ✅ Compile the project
- ✅ Build the classpath

**Watch the bottom-right corner** - you'll see progress like:
```
Building workspace... (X%)
```

Wait until it says "Build complete" or no activity indicator.

---

### Step 4: Update Maven Project (If Needed)

If you see errors, update the Maven project:

1. **Right-click on `javabackend` project** in Package Explorer
2. Select **Maven → Update Project...**
3. Check:
   - ✅ Force Update of Snapshots/Releases
   - ✅ Update project configuration from pom.xml
4. Click **OK**

Wait for it to finish updating.

---

### Step 5: Fix Lombok (If Needed)

If you see Lombok errors:

1. **Install Lombok Plugin:**
   - Help → Eclipse Marketplace
   - Search: "Lombok"
   - Install "Lombok" by Project Lombok
   - Restart Eclipse

2. **Enable Annotation Processing:**
   - Right-click `javabackend` project → Properties
   - Java Compiler → Annotation Processing
   - Check: ✅ Enable annotation processing
   - Click Apply and Close

---

### Step 6: Run the Application

1. Navigate to: `src/main/java/com/lipika/LipikaApplication.java`
2. **Right-click on `LipikaApplication.java`**
3. Select **Run As → Java Application**

---

### Step 7: Verify It's Running

Look for this in the Console:
```
Started LipikaApplication in X.XXX seconds
```

Then test in browser:
- http://localhost:8080/api/health

---

## Troubleshooting

### Issue: "Project has errors"

**Check:**
1. Java version - must be Java 17+
   - Window → Preferences → Java → Installed JREs
   - Add Java 17 if needed

2. Maven dependencies - update project
   - Right-click project → Maven → Update Project

3. Lombok - install plugin (see Step 5 above)

---

### Issue: "Port 8080 already in use"

**Solution:**
1. Stop the existing application in Eclipse Console
2. Or kill the process:
   ```powershell
   netstat -ano | findstr :8080
   taskkill /PID <PID> /F
   ```

---

### Issue: "Class file not found"

**Solution:**
1. Project → Clean...
2. Select `javabackend`
3. Click Clean
4. Wait for rebuild
5. Try running again

---

### Issue: "Java version mismatch"

**Solution:**
1. Right-click project → Properties
2. Java Build Path → Libraries
3. Remove old JRE, add Java 17
4. Project Facets → Java → Set to 17

---

## Expected Console Output

When running successfully, you should see:

```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::                (v3.2.0)

2024-xx-xx INFO ... Starting LipikaApplication
...
2024-xx-xx INFO ... Started LipikaApplication in X.XXX seconds
```

---

## Success Indicators

✅ Console shows "Started LipikaApplication"  
✅ No red errors in Problems view  
✅ Browser shows JSON when visiting http://localhost:8080/api/health  
✅ Frontend can now connect to backend  

---

**That's it! Eclipse handles everything automatically. No command line Maven needed!** 🎉
