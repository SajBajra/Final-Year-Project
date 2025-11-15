# MySQL Database Setup

This directory contains the database schema and setup instructions for the Lipika OCR System.

## Database Configuration

The database configuration is set in `src/main/resources/application.properties`:

```properties
spring.datasource.url=jdbc:mysql://localhost:3306/lipika_db?useSSL=false&serverTimezone=UTC&allowPublicKeyRetrieval=true&createDatabaseIfNotExist=true
spring.datasource.username=root
spring.datasource.password=root
```

**⚠️ IMPORTANT:** Change the `username` and `password` in `application.properties` to match your MySQL installation.

## Setup Instructions

### 1. Install MySQL

If you don't have MySQL installed:

- **Windows**: Download from [MySQL Downloads](https://dev.mysql.com/downloads/mysql/)
- **macOS**: Use Homebrew: `brew install mysql`
- **Linux**: Use package manager:
  - Ubuntu/Debian: `sudo apt-get install mysql-server`
  - CentOS/RHEL: `sudo yum install mysql-server`

### 2. Start MySQL Server

**Windows:**
- MySQL usually starts automatically as a Windows service
- Or use MySQL Workbench to start the server

**macOS/Linux:**
```bash
sudo systemctl start mysql
# or
sudo service mysql start
```

### 3. Create Database and Tables

#### Option A: Using MySQL Command Line

```bash
# Login to MySQL
mysql -u root -p

# Run the schema script
source /path/to/javabackend/database/schema.sql

# Or execute SQL directly
mysql -u root -p < database/schema.sql
```

#### Option B: Using Spring Boot Auto-Update (Recommended for Development)

Spring Boot will automatically create the database and tables if:
- `spring.jpa.hibernate.ddl-auto=update` is set (already configured)
- `createDatabaseIfNotExist=true` is in the connection URL (already configured)

Just start the Spring Boot application and it will create everything automatically.

#### Option C: Using MySQL Workbench

1. Open MySQL Workbench
2. Connect to your MySQL server
3. Open `database/schema.sql`
4. Execute the script

### 4. Verify Database Creation

```bash
mysql -u root -p -e "USE lipika_db; SHOW TABLES;"
```

You should see:
- `ocr_history`
- `system_settings`
- `admin_users` (for future use)

### 5. Update Application Properties

Edit `src/main/resources/application.properties` and update:

```properties
# Change these to match your MySQL installation
spring.datasource.username=your_username
spring.datasource.password=your_password
```

### 6. Test Database Connection

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
1. Check if MySQL server is running
2. Verify the connection URL and port (default: 3306)
3. Check firewall settings

### Access Denied Error

```
Access denied for user 'root'@'localhost'
```

**Solution:**
1. Verify username and password in `application.properties`
2. Create a MySQL user with proper permissions:
   ```sql
   CREATE USER 'lipika_user'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON lipika_db.* TO 'lipika_user'@'localhost';
   FLUSH PRIVILEGES;
   ```
3. Update `application.properties` with the new credentials

### Unknown Database Error

```
Unknown database 'lipika_db'
```

**Solution:**
- Make sure `createDatabaseIfNotExist=true` is in the connection URL
- Or manually create the database: `CREATE DATABASE lipika_db;`

### Character Encoding Issues

The database uses `utf8mb4` encoding to support Devanagari and other Unicode characters. If you see encoding issues:
1. Verify database charset: `SHOW CREATE DATABASE lipika_db;`
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

