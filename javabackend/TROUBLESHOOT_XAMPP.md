# Troubleshooting XAMPP MySQL Connection

## Error: "Access denied for user 'root'@'localhost' (using password: YES)"

This error occurs when Spring Boot tries to authenticate with MySQL but the password doesn't match.

## Solution Steps:

### Step 1: Verify XAMPP MySQL Password

**Option A: Check if MySQL has no password (XAMPP default)**

1. Open **XAMPP Control Panel**
2. Click **"Admin"** button next to MySQL (opens phpMyAdmin)
3. Or go to: `http://localhost/phpmyadmin`
4. Try to login with:
   - Username: `root`
   - Password: *(leave empty)*

**If login succeeds with empty password:**
- Your MySQL has no password (XAMPP default)
- Use the configuration below (no password line)

**If login fails with empty password:**
- Your MySQL has a password set
- You need to find/reset the password

### Step 2: Reset XAMPP MySQL Password (if needed)

If you don't know the password, you can reset it:

**Method 1: Reset via XAMPP Shell**
1. Open **XAMPP Control Panel**
2. Click **"Shell"** button
3. Run:
   ```bash
   mysql.exe -u root -e "ALTER USER 'root'@'localhost' IDENTIFIED BY '';"
   ```
   This sets the password to empty (XAMPP default)

**Method 2: Reset via MySQL Command**
1. Stop MySQL in XAMPP Control Panel
2. Open Command Prompt as Administrator
3. Navigate to XAMPP MySQL:
   ```bash
   cd C:\xampp\mysql\bin
   ```
4. Start MySQL in safe mode (skip grant tables):
   ```bash
   mysqld.exe --skip-grant-tables --console
   ```
5. Open another Command Prompt window
6. Connect without password:
   ```bash
   mysql.exe -u root
   ```
7. Reset password:
   ```sql
   USE mysql;
   UPDATE user SET authentication_string='' WHERE User='root';
   FLUSH PRIVILEGES;
   EXIT;
   ```
8. Stop the safe mode MySQL (Ctrl+C in the first window)
9. Restart MySQL normally in XAMPP

### Step 3: Update Application Configuration

**If MySQL has NO password (XAMPP default):**

Remove or comment out the password line in `application.properties`:
```properties
spring.datasource.username=root
# spring.datasource.password=  # Remove this line or comment it out
```

**If MySQL has a password:**

Set the password in `application.properties`:
```properties
spring.datasource.username=root
spring.datasource.password=your_actual_password
```

### Step 4: Alternative - Test Connection Manually

Test the connection manually using MySQL command line:

```bash
# Navigate to XAMPP MySQL
cd C:\xampp\mysql\bin

# Try to connect (no password)
mysql.exe -u root

# If that fails, try with password
mysql.exe -u root -p
# Then enter your password when prompted
```

### Step 5: Verify Database Exists

Make sure the `lipika` database exists:

**Via phpMyAdmin:**
1. Go to `http://localhost/phpmyadmin`
2. Check if `lipika` database exists in the left sidebar
3. If not, create it:
   - Click "New"
   - Database name: `lipika`
   - Collation: `utf8mb4_unicode_ci`
   - Click "Create"

**Via MySQL Command:**
```sql
CREATE DATABASE IF NOT EXISTS lipika CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### Step 6: Restart Spring Boot Application

After updating the configuration:
1. Stop the Spring Boot application
2. Clean and rebuild:
   ```bash
   cd javabackend
   mvn clean install
   ```
3. Start the application again

## Quick Fix Checklist

- [ ] XAMPP MySQL is running (green in XAMPP Control Panel)
- [ ] Verified MySQL password (empty or known password)
- [ ] Updated `application.properties` with correct password (or removed password line)
- [ ] `lipika` database exists in phpMyAdmin
- [ ] Restarted Spring Boot application after configuration change

## Common Issues

**Issue:** "Communications link failure"
- **Solution:** Make sure MySQL is running in XAMPP Control Panel

**Issue:** "Unknown database 'lipika'"
- **Solution:** Create the database in phpMyAdmin or run `schema.sql`

**Issue:** "Access denied" even with correct password
- **Solution:** Reset MySQL password using methods above

## Still Not Working?

1. Check XAMPP MySQL error logs: `C:\xampp\mysql\data\mysql_error.log`
2. Verify MySQL port (default: 3306) in XAMPP config
3. Try connecting with MySQL Workbench or phpMyAdmin to verify credentials
4. Check if Windows Firewall is blocking localhost connections

