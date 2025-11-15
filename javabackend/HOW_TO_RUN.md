# How to Run Java Backend

## Option 1: Using IDE (Recommended)

### IntelliJ IDEA:
1. **Open the project:**
   - `File` → `Open` → Select `javabackend` folder
   
2. **Wait for Maven to sync:**
   - IntelliJ will automatically detect `pom.xml` and download dependencies
   - Wait for "Maven sync" to complete in the bottom toolbar
   
3. **Build the project:**
   - `Build` → `Build Project` (or press `Ctrl+F9`)
   - Or right-click on `pom.xml` → `Maven` → `Reload Project`
   
4. **Run the application:**
   - Find `LipikaApplication.java` in `src/main/java/com/lipika/`
   - Right-click on the file → `Run 'LipikaApplication.main()'`
   - Or click the green play button next to the `main` method

### Eclipse:
1. **Import the project:**
   - `File` → `Import` → `Existing Maven Projects`
   - Select `javabackend` folder
   
2. **Build:**
   - Right-click project → `Maven` → `Update Project`
   - Right-click project → `Build Project`
   
3. **Run:**
   - Right-click `LipikaApplication.java` → `Run As` → `Java Application`

### VS Code:
1. **Open folder:**
   - `File` → `Open Folder` → Select `javabackend` folder
   
2. **Install extensions:**
   - Install "Extension Pack for Java" if not already installed
   
3. **Build:**
   - Open command palette (`Ctrl+Shift+P`)
   - Type "Java: Clean Java Language Server Workspace"
   - Then "Java: Restart Language Server"
   
4. **Run:**
   - Click the play button above `main` method in `LipikaApplication.java`

## Option 2: Using Command Line (if Maven is installed)

```bash
cd javabackend

# Clean and build
mvn clean compile

# Run the application
mvn spring-boot:run
```

## Option 3: Using Gradle Wrapper (if available)

```bash
cd javabackend

./gradlew build
./gradlew bootRun
```

## Troubleshooting

### If you get "ClassNotFoundException":
1. **Clean and rebuild:**
   - In IDE: `Build` → `Clean` then `Build` → `Build Project`
   - In command line: `mvn clean install`
   
2. **Refresh dependencies:**
   - In IntelliJ: Right-click `pom.xml` → `Maven` → `Reload Project`
   - In Eclipse: Right-click project → `Maven` → `Update Project` → Check "Force Update"

3. **Check Java version:**
   - Make sure Java 17 is installed
   - Check: `java -version`
   - Should show version 17 or higher

### If you get "Could not find or load main class":
1. Make sure `LipikaApplication.java` exists at:
   - `src/main/java/com/lipika/LipikaApplication.java`
   
2. Rebuild the project completely:
   - Delete `target` folder if it exists
   - Clean and rebuild

3. Verify the main class in `pom.xml`:
   ```xml
   <mainClass>com.lipika.LipikaApplication</mainClass>
   ```

### If application.yml has errors:
1. The file should have only ONE `spring:` key
2. Make sure UTF-8 encoding is configured
3. Check for duplicate keys

## Verifying Setup

After successful build, you should see:
- `target/classes/com/lipika/LipikaApplication.class` exists
- Application starts on `http://localhost:8080`
- Health endpoint works: `http://localhost:8080/api/health`

## Expected Output

When running successfully, you should see:
```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::                (v3.2.0)

... (more logs) ...

Started LipikaApplication in X.XXX seconds
```

