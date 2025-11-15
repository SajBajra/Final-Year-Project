# How to Build and Run Java Backend

## Prerequisites
- Java 17 or higher installed
- Maven installed (or use IDE's built-in Maven)

## Quick Start

### Option 1: Using IDE (Easiest)

**IntelliJ IDEA:**
1. Open `javabackend` folder in IntelliJ
2. Wait for Maven sync (bottom-right corner)
3. Right-click `pom.xml` → `Maven` → `Reload Project`
4. Click `Build` → `Rebuild Project` (or press `Ctrl+Shift+F9`)
5. After build completes, find `LipikaApplication.java`
6. Right-click → `Run 'LipikaApplication.main()'`

**Eclipse:**
1. `File` → `Import` → `Existing Maven Projects`
2. Select `javabackend` folder
3. Right-click project → `Maven` → `Update Project`
4. `Project` → `Build Project`
5. Right-click `LipikaApplication.java` → `Run As` → `Java Application`

**VS Code:**
1. Install "Extension Pack for Java"
2. Open `javabackend` folder
3. `Ctrl+Shift+P` → "Java: Clean Java Language Server Workspace"
4. `Ctrl+Shift+P` → "Java: Rebuild Projects"
5. Click ▶️ above `main` method in `LipikaApplication.java`

### Option 2: Using Command Line

**Windows (PowerShell):**
```powershell
cd javabackend
mvn clean compile
mvn spring-boot:run
```

**Windows (Command Prompt):**
```cmd
cd javabackend
mvn clean compile
mvn spring-boot:run
```

**Linux/Mac:**
```bash
cd javabackend
mvn clean compile
mvn spring-boot:run
```

### Option 3: Using Build Script

**Windows:**
```cmd
cd javabackend
build.bat
```

## Verify Build

After building, check that these files exist:
- `target/classes/com/lipika/LipikaApplication.class`
- `target/classes/com/lipika/service/impl/OCRServiceImpl.class`

## Troubleshooting

### Error: "Could not find or load main class"
- **Solution:** Build the project first using one of the methods above
- The project needs to be compiled before running

### Error: "mvn command not found"
- **Solution:** Install Maven or use IDE's built-in Maven
- Download Maven: https://maven.apache.org/download.cgi
- Or use IDE: IntelliJ/Eclipse have built-in Maven

### Error: "Java version error"
- **Solution:** Install Java 17 or higher
- Check version: `java -version`
- Should show version 17 or higher

### Build Errors
1. Clean the build: `mvn clean`
2. Update dependencies: `mvn dependency:resolve`
3. Rebuild: `mvn compile`
4. If using IDE: Invalidate caches and rebuild

## Expected Output

When running successfully:
```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 :: Spring Boot ::                (v3.2.0)

... (more logs) ...

Started LipikaApplication in X.XXX seconds
```

## Next Steps

After successful build and run:
1. Verify service is running on `http://localhost:8080`
2. Check health: `http://localhost:8080/api/health`
3. Start Python OCR service on `http://localhost:5000`
4. Start frontend: `cd frontend && npm run dev`

