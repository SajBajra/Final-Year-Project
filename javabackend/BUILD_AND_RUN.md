# 🚀 Build and Run Guide - Lipika Backend

## Quick Start

### Prerequisites Check

```bash
# Check Java version (must be 17+)
java -version

# Check Maven version (must be 3.6+)
mvn --version
```

If either command fails, install the missing tool first.

---

## Building the Project

### Step 1: Navigate to Backend Directory

```bash
cd javabackend
```

### Step 2: Build the Project

```bash
mvn clean install
```

This will:
- Download all dependencies
- Compile Java source files
- Run tests
- Create JAR file in `target/` directory

**Expected Output:**
```
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
```

---

## Running the Application

### Option 1: Using Maven (Recommended for Development)

```bash
mvn spring-boot:run
```

### Option 2: Using JAR File

```bash
java -jar target/lipika-backend-1.0.0.jar
```

### Option 3: Using IDE

1. Open `javabackend` folder in IntelliJ IDEA or Eclipse
2. Wait for Maven dependencies to download
3. Run `LipikaApplication.java` as a Java application

---

## Verifying It's Running

Once started, you should see:
```
Started LipikaApplication in X.XXX seconds
```

Then test the health endpoint:
```bash
curl http://localhost:8080/api/health
```

Expected response:
```json
{
  "success": true,
  "message": "Service is healthy",
  "data": {
    "status": "UP",
    "timestamp": "...",
    "service": "Lipika Backend - Presenter Layer",
    "version": "1.0.0"
  }
}
```

---

## Troubleshooting

### Error: "Could not find or load main class com.lipika.LipikaApplication"

**Cause:** Project hasn't been compiled yet.

**Fix:**
```bash
cd javabackend
mvn clean install
mvn spring-boot:run
```

### Error: "mvn: command not found"

**Cause:** Maven is not installed or not in PATH.

**Fix - Install Maven:**

**Windows (Winget):**
```powershell
winget install Apache.Maven
```

**Windows (Chocolatey):**
```powershell
choco install maven
```

**Manual Installation:**
1. Download from: https://maven.apache.org/download.cgi
2. Extract to a folder (e.g., `C:\Program Files\Apache\maven`)
3. Add `bin` folder to PATH environment variable
4. Restart terminal

**Verify:**
```bash
mvn --version
```

### Error: "Java version X is not supported"

**Cause:** Java version is too old (need Java 17+).

**Fix:** Install Java 17 or newer.

**Check current version:**
```bash
java -version
```

**Install Java 17:**
- Download from: https://adoptium.net/
- Or use package manager:
  ```powershell
  winget install EclipseAdoptium.Temurin.17.JDK
  ```

### Error: "Port 8080 already in use"

**Cause:** Another application is using port 8080.

**Fix Option 1:** Stop the other application

**Fix Option 2:** Change port in `application.properties`:
```properties
server.port=8081
```

---

## Development Workflow

1. **Make code changes**
2. **Rebuild (if needed):**
   ```bash
   mvn clean install
   ```
3. **Run:**
   ```bash
   mvn spring-boot:run
   ```

**Note:** If using Spring Boot DevTools, the application will auto-reload on code changes (no need to restart manually).

---

## Building for Production

### Create Executable JAR

```bash
mvn clean package
```

The JAR file will be at: `target/lipika-backend-1.0.0.jar`

### Run Production JAR

```bash
java -jar target/lipika-backend-1.0.0.jar
```

### Run with Custom Configuration

```bash
java -jar target/lipika-backend-1.0.0.jar --server.port=8081 --ocr.service.url=http://localhost:5000
```

---

## Need Help?

If you're still having issues:

1. Check Java version: `java -version` (must be 17+)
2. Check Maven version: `mvn --version` (must be 3.6+)
3. Clean and rebuild: `mvn clean install`
4. Check for compilation errors in the output
5. Verify all files are in the correct directory structure
