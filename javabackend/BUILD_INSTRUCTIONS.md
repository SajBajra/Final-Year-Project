# Quick Build Instructions - Fix "ClassNotFoundException"

## The Problem
```
Error: Could not find or load main class com.lipika.LipikaApplication
Caused by: java.lang.ClassNotFoundException: com.lipika.LipikaApplication
```

This means the project hasn't been compiled yet. You need to build it first.

## Solution: Build the Project

### Option 1: Using IntelliJ IDEA (Recommended)

1. **Open the project:**
   - Open IntelliJ IDEA
   - `File` → `Open` → Select `E:\Cllz\FYP\javabackend` folder

2. **Wait for Maven sync:**
   - IntelliJ will automatically detect the Maven project
   - Wait for "Maven sync" to complete (bottom-right corner)

3. **Reload Maven project:**
   - Right-click on `pom.xml` in the project explorer
   - Select `Maven` → `Reload Project`

4. **Build the project:**
   - Go to `Build` → `Rebuild Project` (or press `Ctrl+Shift+F9`)
   - Wait for build to complete (check bottom status bar)

5. **Verify build:**
   - Check if `target/classes/com/lipika/LipikaApplication.class` exists
   - If it exists, the build was successful!

6. **Run the application:**
   - Find `LipikaApplication.java` in `src/main/java/com/lipika/`
   - Right-click on the class → `Run 'LipikaApplication.main()'`

### Option 2: Using Eclipse

1. **Import the project:**
   - `File` → `Import` → `Existing Maven Projects`
   - Select `E:\Cllz\FYP\javabackend` folder
   - Click `Finish`

2. **Update Maven project:**
   - Right-click on the project in Package Explorer
   - `Maven` → `Update Project...`
   - Check "Force Update of Snapshots/Releases"
   - Click `OK`

3. **Build the project:**
   - `Project` → `Clean...` → Select your project → `Clean`
   - `Project` → `Build Project` (or press `Ctrl+B`)

4. **Verify build:**
   - Check if `target/classes/com/lipika/LipikaApplication.class` exists

5. **Run the application:**
   - Right-click `LipikaApplication.java`
   - `Run As` → `Java Application`

### Option 3: Using VS Code

1. **Install Java extensions:**
   - Install "Extension Pack for Java" from VS Code marketplace

2. **Open the project:**
   - `File` → `Open Folder` → Select `E:\Cllz\FYP\javabackend`

3. **Clean and rebuild:**
   - Press `Ctrl+Shift+P`
   - Type "Java: Clean Java Language Server Workspace"
   - Press `Enter`
   - Press `Ctrl+Shift+P` again
   - Type "Java: Rebuild Projects"
   - Press `Enter`

4. **Run the application:**
   - Open `LipikaApplication.java`
   - Click the ▶️ button above the `main` method

### Option 4: Using Command Line (if Maven is installed)

**If Maven is installed and in PATH:**

```bash
cd E:\Cllz\FYP\javabackend
mvn clean compile
mvn spring-boot:run
```

**If Maven is NOT installed:**

You can use the build script (if you have Maven installed separately):

```bash
cd E:\Cllz\FYP\javabackend
build.bat
```

## Verify Build Success

After building, check that these files exist:
- ✅ `target/classes/com/lipika/LipikaApplication.class`
- ✅ `target/classes/com/lipika/model/OCRHistory.class`
- ✅ `target/classes/com/lipika/service/impl/AdminServiceImpl.class`

If these `.class` files exist, the build was successful!

## What to Do After Successful Build

1. **Make sure XAMPP MySQL is running:**
   - Open XAMPP Control Panel
   - Start MySQL (should be green)

2. **Verify database exists:**
   - Go to `http://localhost/phpmyadmin`
   - Check if `lipika` database exists
   - If not, create it or run `database/schema.sql`

3. **Run the application:**
   - Use your IDE to run `LipikaApplication`
   - Or use command: `mvn spring-boot:run`

4. **Check for errors:**
   - Application should start on `http://localhost:8080`
   - Watch the console for any database connection errors

## Common Issues

**"Maven sync failed":**
- Check your internet connection (Maven downloads dependencies)
- Try: `File` → `Invalidate Caches / Restart` (IntelliJ)

**"Build failed":**
- Check Java version: `java -version` (should be 17 or higher)
- Check if you have JDK 17 installed

**"Still getting ClassNotFoundException after build":**
- Make sure you're running from the IDE, not trying to run the `.java` file directly
- Clean the project and rebuild again
- Check if `target/classes` folder has the compiled `.class` files

## Next Steps

Once the application starts successfully, you should see:
```
Started LipikaApplication in X.XXX seconds
```

Then you can:
- Access the API at `http://localhost:8080`
- Test endpoints: `http://localhost:8080/api/health`
- Start the frontend and connect to the backend

