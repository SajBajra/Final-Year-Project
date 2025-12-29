# Troubleshooting Compilation Errors

## Common Errors and Solutions

### Error 1: "log cannot be resolved" or "@Slf4j not working"

**Problem:** Lombok annotation processing is not enabled in your IDE.

**Solution for Spring Tool Suite (STS) / Eclipse:**

1. **Install Lombok Plugin:**
   - Download Lombok jar from: https://projectlombok.org/download
   - Double-click the `lombok.jar` file
   - Select your STS/Eclipse installation directory
   - Click "Install/Update"
   - Restart STS/Eclipse

2. **Enable Annotation Processing:**
   - Right-click project → Properties
   - Java Compiler → Annotation Processing
   - Check "Enable annotation processing"
   - Click Apply and Close

3. **Update Maven Project:**
   - Right-click project → Maven → Update Project
   - Check "Force Update of Snapshots/Releases"
   - Click OK

4. **Clean and Rebuild:**
   - Project → Clean → Select your project → Clean
   - Project → Build Project

**Solution for IntelliJ IDEA:**

1. **Install Lombok Plugin:**
   - File → Settings → Plugins
   - Search for "Lombok"
   - Install the Lombok plugin
   - Restart IntelliJ

2. **Enable Annotation Processing:**
   - File → Settings → Build, Execution, Deployment → Compiler → Annotation Processors
   - Check "Enable annotation processing"
   - Click Apply and OK

3. **Invalidate Caches:**
   - File → Invalidate Caches / Restart
   - Select "Invalidate and Restart"

---

### Error 2: "Cannot infer type arguments for ApiResponse<T>"

**Problem:** IDE not recognizing generic types properly.

**Solutions:**

1. **Update Maven Dependencies:**
   ```bash
   cd javabackend
   mvn clean install
   ```

2. **In STS/Eclipse:**
   - Right-click project → Maven → Update Project
   - Check "Force Update of Snapshots/Releases"
   - Project → Clean → Build

3. **In IntelliJ:**
   - File → Invalidate Caches / Restart
   - Right-click on `pom.xml` → Maven → Reload Project

4. **Check Java Version:**
   - Make sure you're using Java 17 or higher
   - Right-click project → Properties → Java Build Path → Libraries
   - Verify Java version is 17+

---

### Error 3: "Package does not exist" or "Cannot resolve symbol"

**Problem:** Maven dependencies not downloaded.

**Solutions:**

1. **Download Dependencies:**
   ```bash
   cd javabackend
   mvn dependency:resolve
   ```

2. **In STS/Eclipse:**
   - Right-click project → Maven → Update Project
   - Wait for dependencies to download

3. **In IntelliJ:**
   - Right-click `pom.xml` → Maven → Reload Project
   - Wait for dependencies to download

4. **Check Internet Connection:**
   - Maven needs internet to download dependencies
   - Check if you're behind a proxy

---

### Error 4: Multiple compilation errors (100+ errors)

**Problem:** Usually means Lombok or Maven dependencies aren't set up correctly.

**Step-by-Step Fix:**

1. **First, ensure Lombok is installed in IDE** (see Error 1)

2. **Clean Maven:**
   ```bash
   cd javabackend
   mvn clean
   ```

3. **Update Maven Project:**
   - STS/Eclipse: Right-click → Maven → Update Project
   - IntelliJ: Right-click `pom.xml` → Maven → Reload Project

4. **Rebuild:**
   ```bash
   mvn compile
   ```

5. **If still errors, check:**
   - Java version (must be 17+)
   - Maven is installed and in PATH
   - IDE has correct JDK configured

---

## Quick Fix Checklist

✅ **Lombok Plugin Installed?**
- STS/Eclipse: Check if lombok.jar is installed
- IntelliJ: Check if Lombok plugin is installed

✅ **Annotation Processing Enabled?**
- STS/Eclipse: Properties → Java Compiler → Annotation Processing
- IntelliJ: Settings → Annotation Processors

✅ **Maven Dependencies Downloaded?**
- Check `.m2/repository` folder exists
- Run `mvn dependency:resolve`

✅ **Java Version Correct?**
- Must be Java 17 or higher
- Check: `java -version`

✅ **Project Cleaned?**
- Project → Clean → Build

---

## Manual Fix for "log cannot be resolved"

If Lombok still doesn't work, you can manually add logging:

**Instead of:**
```java
@Slf4j
public class MyClass {
    public void method() {
        log.info("Message");
    }
}
```

**Use:**
```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MyClass {
    private static final Logger log = LoggerFactory.getLogger(MyClass.class);
    
    public void method() {
        log.info("Message");
    }
}
```

---

## Verify Setup

After fixing, verify everything works:

1. **Check Lombok:**
   ```bash
   mvn compile
   ```
   Should compile without errors

2. **Check IDE:**
   - Open any class with `@Slf4j`
   - Type `log.` - should show autocomplete
   - If not, Lombok isn't working

3. **Run Application:**
   - Right-click `LipikaApplication.java`
   - Run As → Spring Boot App
   - Should start without errors

---

## Still Having Issues?

1. **Check pom.xml:**
   - Make sure Lombok dependency is present
   - Version should match Spring Boot version

2. **Check IDE Logs:**
   - STS/Eclipse: Window → Show View → Error Log
   - IntelliJ: Help → Show Log in Explorer

3. **Re-import Project:**
   - Delete `.project`, `.classpath`, `.settings` (STS/Eclipse)
   - Delete `.idea` folder (IntelliJ)
   - Re-import project

4. **Use Command Line:**
   ```bash
   cd javabackend
   mvn clean compile
   ```
   If this works, it's an IDE configuration issue.

