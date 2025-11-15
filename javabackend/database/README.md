# MySQL Database Setup

This directory contains the database schema and setup instructions for the Lipika OCR System.

## Database Configuration (XAMPP)

The database configuration is set in `src/main/resources/application.properties`:

```properties
spring.datasource.url=jdbc:mysql://localhost:3306/lipika?useSSL=false&serverTimezone=UTC&allowPublicKeyRetrieval=true
spring.datasource.username=root
spring.datasource.password=
```

**⚠️ NOTE:** This configuration is set for XAMPP's default MySQL setup (no password for root user). If you've changed your XAMPP MySQL root password, update it in `application.properties`.

## Setup Instructions (XAMPP)

### 1. Start XAMPP

1. Open **XAMPP Control Panel**
2. Start **Apache** (if needed for other services)
3. Start **MySQL** by clicking the "Start" button

### 2. Create Database and Tables

#### Option A: Using phpMyAdmin (Recommended for XAMPP)

1. Open your browser and go to: `http://localhost/phpmyadmin`
2. Click on **"SQL"** tab at the top
3. Copy and paste the entire contents of `database/schema.sql`
4. Click **"Go"** to execute

The script will:
- Create the `lipika` database
- Create all necessary tables
- Insert default settings

#### Option B: Using MySQL Command Line

```bash
# Navigate to XAMPP MySQL bin directory
cd C:\xampp\mysql\bin

# Login to MySQL (no password for default XAMPP setup)
mysql.exe -u root

# Or if you have MySQL in PATH
mysql -u root

# Run the schema script
source E:\Cllz\FYP\javabackend\database\schema.sql

# Or execute SQL directly
mysql.exe -u root < E:\Cllz\FYP\javabackend\database\schema.sql
```

#### Option C: Using Spring Boot Auto-Update (Recommended for Development)

Spring Boot will automatically create tables if:
- `spring.jpa.hibernate.ddl-auto=update` is set (already configured)
- Database `lipika` already exists

**Steps:**
1. Create database manually in phpMyAdmin: `CREATE DATABASE lipika;`
2. Start Spring Boot application
3. It will automatically create all tables

#### Option D: Manual Creation via phpMyAdmin

1. Open phpMyAdmin: `http://localhost/phpmyadmin`
2. Click **"New"** to create a new database
3. Enter database name: `lipika`
4. Select collation: `utf8mb4_unicode_ci`
5. Click **"Create"**
6. Select the `lipika` database
7. Click **"SQL"** tab and paste the table creation queries from `schema.sql`

### 3. Verify Database Creation

In phpMyAdmin or MySQL command line:

```sql
USE lipika;
SHOW TABLES;
```

You should see:
- `ocr_history`
- `system_settings`
- `admin_users` (for future use)

### 4. Verify Application Properties

The configuration is already set for XAMPP default:
```properties
spring.datasource.url=jdbc:mysql://localhost:3306/lipika
spring.datasource.username=root
spring.datasource.password=
```

**⚠️ If you changed XAMPP MySQL root password**, update `application.properties`:
```properties
spring.datasource.password=your_password
```

### 5. Test Database Connection

Start the Spring Boot application. If the database connection is successful, you should see no errors in the logs. If there are errors, check:

1. MySQL server is running
2. Username and password are correct
3. Database exists (or `createDatabaseIfNotExist=true` is set)
4. MySQL port (default: 3306) is accessible

## Database Schema

### ocr_history Table

Stores all OCR recognition history:
- `id`: Auto-increment primary key
- `image_filename`: Original image filename
- `recognized_text`: Recognized Devanagari text
- `character_count`: Number of characters recognized
- `confidence`: Average confidence score (0.0 - 1.0)
- `timestamp`: When the OCR was performed
- `language`: Language of recognized text (default: "devanagari")

### system_settings Table

Stores system configuration (for future use):
- `setting_key`: Unique setting identifier
- `setting_value`: Setting value
- `description`: Setting description

### admin_users Table

Stores admin user accounts (for future use when implementing proper authentication):
- `id`: Auto-increment primary key
- `username`: Admin username
- `password_hash`: Hashed password
- `email`: Admin email
- `is_active`: Whether account is active

## Migration from In-Memory Storage

When you first run the application with MySQL:
- All existing in-memory data will be lost (this is expected)
- New OCR operations will be saved to the database
- Data will persist across application restarts

## Troubleshooting

### Connection Refused Error

```
com.mysql.cj.jdbc.exceptions.CommunicationsException: Communications link failure
```

**Solution:**
1. **Check XAMPP MySQL is running**: Open XAMPP Control Panel and ensure MySQL is started (green status)
2. Verify MySQL port (default: 3306) - check XAMPP Control Panel
3. Check Windows Firewall if blocking localhost connections

### Access Denied Error

```
Access denied for user 'root'@'localhost'
```

**Solution:**
1. **XAMPP Default**: XAMPP MySQL root user has no password by default - leave `password=` empty in `application.properties`
2. **If you set a password**: Update `spring.datasource.password=your_password` in `application.properties`
3. **Reset XAMPP MySQL password** (if needed):
   - Stop MySQL in XAMPP
   - Delete `mysql/data/mysql` folder (backup first!)
   - Restart MySQL in XAMPP
   - Default: no password for root

### Unknown Database Error

```
Unknown database 'lipika'
```

**Solution:**
- **Create database manually** in phpMyAdmin:
  1. Go to `http://localhost/phpmyadmin`
  2. Click "New" → Enter `lipika` → Select `utf8mb4_unicode_ci` → Click "Create"
- **Or run schema.sql** which creates the database automatically

### Character Encoding Issues

The database uses `utf8mb4` encoding to support Devanagari and other Unicode characters. If you see encoding issues:
1. Verify database charset: `SHOW CREATE DATABASE lipika;`
2. Verify table charset: `SHOW CREATE TABLE ocr_history;`

## Production Recommendations

For production deployment:

1. **Change Default Password**: Update admin password and MySQL root password
2. **Use Connection Pooling**: HikariCP is already configured
3. **Enable SSL**: Update connection URL to use SSL
4. **Backup Strategy**: Set up regular database backups
5. **Monitoring**: Monitor database performance and connections
6. **Indexes**: Additional indexes may be needed based on query patterns
7. **User Permissions**: Create a dedicated database user with minimal required permissions

## Notes

- The application uses JPA/Hibernate with `ddl-auto=update`, which automatically updates the schema
- For production, consider using `ddl-auto=validate` or migration tools like Flyway/Liquibase
- The `ocr_history` table uses UTF-8MB4 encoding to properly store Devanagari characters
- **XAMPP Note**: Default MySQL root user has no password. If you've changed it, update `application.properties`
- **XAMPP MySQL Port**: Default is 3306. If you changed it in XAMPP config, update the connection URL

