# Java Backend Setup Instructions

## Quick Setup for Spring Suite / IntelliJ IDEA

### Prerequisites
- Java JDK 17 or higher
- Maven (usually comes with Spring Suite/IntelliJ)
- MySQL/XAMPP (for database)
- Python OCR service running on port 5000

---

## Step 1: Import Project

### In Spring Tool Suite (STS) / Eclipse:
1. File → Import → Maven → Existing Maven Projects
2. Browse to the `javabackend` folder
3. Click Finish
4. Wait for Maven to download dependencies

### In IntelliJ IDEA:
1. File → Open
2. Select the `javabackend` folder
3. IntelliJ will automatically detect it as a Maven project
4. Wait for dependencies to download

---

## Step 2: Configure Database

### Option A: Using XAMPP (Recommended for Windows)
1. Start XAMPP Control Panel
2. Start MySQL service
3. Open phpMyAdmin (http://localhost/phpmyadmin)
4. Create database: `lipika`
5. The database schema will be created automatically on first run (via `DatabaseInitializationService`)

### Option B: Using MySQL directly
1. Start MySQL service
2. Create database:
   ```sql
   CREATE DATABASE lipika CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   ```

### Update Database Configuration

Edit `src/main/resources/application.properties`:

```properties
# Database Configuration
spring.datasource.url=jdbc:mysql://localhost:3306/lipika?useSSL=false&serverTimezone=UTC&allowPublicKeyRetrieval=true
spring.datasource.username=root
spring.datasource.password=your_password_here
```

**Important:** Change `your_password_here` to your MySQL root password (or leave empty if no password).

---

## Step 3: Configure Admin User (Optional)

The default admin user is created automatically. To change it, edit `src/main/resources/application.properties`:

```properties
# Admin User Configuration (Default Admin Credentials)
admin.default.username=admin
admin.default.password=admin123
admin.default.email=admin@lipika.com
```

**Security Note:** Change the default password after first login!

---

## Step 4: Run the Application

### In Spring Tool Suite / Eclipse:
1. Right-click on the project
2. Run As → Spring Boot App
   OR
3. Find `LipikaApplication.java` (main class)
4. Right-click → Run As → Java Application

### In IntelliJ IDEA:
1. Find `LipikaApplication.java` in `src/main/java/com/lipika/`
2. Right-click → Run 'LipikaApplication'
   OR
3. Use the green play button next to the main method

### Using Maven Command Line:
```bash
cd javabackend
mvn spring-boot:run
```

---

## Step 5: Verify It's Running

1. Check console for: "Started LipikaApplication"
2. Open browser: http://localhost:8080/api/health
3. You should see a health check response

---

## Configuration Files

### Main Configuration: `application.properties`
- Database connection settings
- Server port (default: 8080)
- OCR service URL (default: http://localhost:5000)
- Admin user credentials

### Important Settings:
```properties
# Server Port
server.port=8080

# OCR Service URL (make sure Python service is running)
lipika.ocr.service.url=http://localhost:5000

# Database
spring.datasource.url=jdbc:mysql://localhost:3306/lipika
spring.datasource.username=root
spring.datasource.password=
```

---

## Troubleshooting

### Issue: "Cannot connect to database"
**Solution:**
- Make sure MySQL/XAMPP is running
- Check database name is `lipika`
- Verify username/password in `application.properties`
- Ensure database exists (will be created automatically if using `DatabaseInitializationService`)

### Issue: "Port 8080 already in use"
**Solution:**
- Change port in `application.properties`: `server.port=8081`
- Or stop the application using port 8080

### Issue: "Cannot connect to OCR service"
**Solution:**
- Make sure Python OCR service is running on port 5000
- Check `lipika.ocr.service.url` in `application.properties`
- Verify Python service is accessible: http://localhost:5000/health

### Issue: "Maven dependencies not downloading"
**Solution:**
- Check internet connection
- In STS/Eclipse: Right-click project → Maven → Update Project
- In IntelliJ: File → Invalidate Caches / Restart

---

## Default Admin Credentials

After first run, you can login with:
- **URL:** http://localhost:8080/api/admin/auth/login
- **Username:** admin
- **Password:** admin123
- **Email:** admin@lipika.com

**IMPORTANT:** Change the password immediately after first login!

---

## Project Structure

```
javabackend/
├── src/
│   ├── main/
│   │   ├── java/com/lipika/
│   │   │   ├── controller/     # REST API endpoints
│   │   │   ├── service/        # Business logic
│   │   │   ├── model/          # Entity classes
│   │   │   ├── repository/     # Data access
│   │   │   ├── config/         # Configuration classes
│   │   │   └── dto/            # Data transfer objects
│   │   └── resources/
│   │       └── application.properties  # Configuration
│   └── test/                   # Unit tests
└── pom.xml                     # Maven dependencies
```

---

## API Endpoints

Once running, the backend provides:
- `POST /api/admin/auth/login` - Admin login
- `POST /api/ocr/recognize` - OCR recognition
- `GET /api/admin/dashboard/stats` - Dashboard statistics
- `GET /api/admin/ocr-history` - OCR history
- And more...

See the controller classes for full API documentation.

