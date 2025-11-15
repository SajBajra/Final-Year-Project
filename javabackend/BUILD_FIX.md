# Fix: ClassNotFoundException - Build the Project First

## The Problem
The error "Could not find or load main class com.lipika.LipikaApplication" means the project hasn't been compiled yet. The `.class` file is missing.

## Quick Fix (Using IDE)

### IntelliJ IDEA:
1. **Open the project:**
   - `File` → `Open` → Select `javabackend` folder
   
2. **Wait for Maven sync:**
   - Bottom-right corner should show "Maven sync" progress
   - Wait until it completes

3. **Build the project:**
   - Press `Ctrl+Shift+F9` (Build Project)
   - OR: `Build` → `Rebuild Project`
   - OR: Right-click `pom.xml` → `Maven` → `Reload Project` → Then build

4. **Verify build:**
   - Check `javabackend/target/classes/com/lipika/LipikaApplication.class` exists
   - Should see green checkmark in IDE

5. **Run:**
   - Find `LipikaApplication.java` in `src/main/java/com/lipika/`
   - Right-click → `Run 'LipikaApplication.main()'`
   - OR: Click the green ▶️ button next to the `main` method

### Eclipse:
1. **Import project:**
   - `File` → `Import` → `Existing Maven Projects`
   - Select `javabackend` folder
   
2. **Update project:**
   - Right-click project → `Maven` → `Update Project`
   - Check "Force Update of Snapshots/Releases"
   - Click OK
   
3. **Build:**
   - `Project` → `Build Project`
   - Check `target/classes` folder for `.class` files

4. **Run:**
   - Right-click `LipikaApplication.java`
   - `Run As` → `Java Application`

### VS Code:
1. **Install extensions:**
   - Install "Extension Pack for Java" (if not installed)

2. **Open folder:**
   - `File` → `Open Folder` → Select `javabackend`

3. **Clean and rebuild:**
   - `Ctrl+Shift+P`
   - Type: "Java: Clean Java Language Server Workspace"
   - Then: "Java: Rebuild Projects"

4. **Run:**
   - Open `LipikaApplication.java`
   - Click ▶️ above `main` method

## After Building

Verify these files exist:
- ✅ `target/classes/com/lipika/LipikaApplication.class`
- ✅ `target/classes/com/lipika/service/impl/OCRServiceImpl.class`
- ✅ `target/classes/com/lipika/config/ApplicationConfig.class`

## Still Getting Error?

1. **Clean build:**
   - Delete `javabackend/target` folder
   - Rebuild project

2. **Check Java version:**
   - Must be Java 17 or higher
   - In IDE: `File` → `Project Structure` → Check Java version

3. **Reload Maven:**
   - Right-click `pom.xml` → `Maven` → `Reload Project`

The project will NOT run until it's built first!

