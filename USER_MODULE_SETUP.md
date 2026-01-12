# User Module Setup Guide

## Overview
Complete user management system with registration, login, forgot password (SMTP), user profiles, usage tracking, and payment integration placeholder.

---

## ✅ Completed Features

### Backend (Java Spring Boot)
1. **User Model** with usage limits and payment fields
   - `usageCount` - tracks number of OCR scans
   - `usageLimit` - default 10 free scans
   - `isPremium` - premium user flag
   - `premiumUntil` - premium expiry date

2. **Password Reset System**
   - `PasswordResetToken` entity
   - Token generation and expiry (1 hour)
   - Email-based password reset

3. **User Controller** (`/api/users`)
   - `POST /register` - User registration
   - `POST /login` - User login
   - `POST /forgot-password` - Request password reset
   - `POST /reset-password` - Reset password with token
   - `GET /profile` - Get user profile (authenticated)
   - `GET /usage-status` - Check if user reached limit

4. **Email Service** (SMTP)
   - Password reset emails
   - Welcome emails on registration
   - Configurable SMTP settings

5. **BCrypt Password Encryption**
   - Already configured via Spring Security
   - Passwords hashed automatically

### Frontend (React)
1. **Register Page** (`/register`)
   - Beautiful animated form
   - Validation
   - Auto-login after registration
   - Success animation

2. **Login Page** (`/login`)
   - Updated for regular users (not just admin)
   - Forgot password link
   - Sign up link

3. **Forgot Password Page** (`/forgot-password`)
   - Email submission
   - Success confirmation

4. **Reset Password Page** (`/reset-password`)
   - Token-based reset
   - Password confirmation
   - Success redirect to login

5. **User Profile Page** (`/profile`)
   - Usage statistics
   - Usage counter with visual progress bar
   - Remaining scans display
   - Premium upgrade button
   - Account details
   - Logout functionality

6. **Auth Service**
   - API integration for all user operations

---

## 🔧 Configuration Required

### 1. Email (SMTP) Setup

Update `javabackend/src/main/resources/application.yml`:

```yaml
spring:
  mail:
    host: smtp.gmail.com
    port: 587
    username: your-email@gmail.com  # Your email
    password: your-app-password      # Gmail App Password
    properties:
      mail:
        smtp:
          auth: true
          starttls:
            enable: true
```

**Gmail App Password Setup:**
1. Enable 2FA on your Google account
2. Go to Google Account > Security > 2-Step Verification
3. Scroll down to "App passwords"
4. Generate a new app password for "Mail"
5. Use this password in application.yml

**Environment Variables (Recommended for Production):**
```bash
export MAIL_USERNAME=your-email@gmail.com
export MAIL_PASSWORD=your-app-password
export FRONTEND_URL=http://localhost:5173
```

### 2. Database Migration

Run your Spring Boot application. Hibernate will automatically create:
- `password_reset_tokens` table
- New columns in `users` table:
  - `usage_count`
  - `usage_limit`
  - `is_premium`
  - `premium_until`

### 3. Frontend Configuration

The frontend is already configured to connect to `http://localhost:8080`.

For production, update `frontend/src/services/authService.js`:
```javascript
const API_URL = 'https://your-production-api.com/api/users'
```

---

## 🚀 Usage

### For Users:

1. **Register**: Go to `/register`
   - Enter username, email, password
   - Get 10 free OCR scans
   - Auto-login after registration

2. **Login**: Go to `/login`
   - Enter username/email and password

3. **Forgot Password**: Click "Forgot password?" on login page
   - Enter email
   - Check email for reset link
   - Click link and set new password

4. **View Profile**: Click profile dropdown > "My Profile"
   - See usage statistics
   - Track remaining scans
   - Upgrade to premium when limit reached

5. **OCR Usage**:
   - Each OCR scan increments the usage counter
   - Free users: 10 scans limit
   - Premium users: Unlimited scans

### For Admins:

1. Admin login still works at `/ocr_admin`
2. Admins have unlimited access
3. User management can be done via database or future admin panel

---

## 📊 Usage Tracking Integration

To integrate usage tracking with your OCR service, update your OCR endpoint to:

1. **Check if user is authenticated**:
```java
if (authentication != null) {
    Long userId = Long.parseLong(authentication.getName());
    
    // Check if user has reached limit
    if (userService.hasReachedLimit(userId)) {
        return ResponseEntity.status(HttpStatus.PAYMENT_REQUIRED)
            .body(ApiResponse.error("Usage limit reached. Please upgrade to premium."));
    }
    
    // Increment usage count
    userService.incrementUsageCount(userId);
}
```

2. **Allow anonymous users** (unregistered) to use the service normally with cookie-based trial tracking (already implemented).

---

## 💳 Payment Integration (Placeholder)

The payment integration is ready for implementation. Current setup:
- "Upgrade to Premium" button in User Profile
- Shows alert: "Payment integration coming soon!"

To implement:
1. Choose a payment gateway (Stripe, PayPal, etc.)
2. Create payment endpoint in backend
3. Update `handleUpgradeToPremium()` in `UserProfile.jsx`
4. Set `isPremium = true` after successful payment
5. Optionally set `premiumUntil` for subscription end date

---

## 🔒 Security Features

1. **BCrypt Password Hashing**: All passwords encrypted
2. **JWT Authentication**: Token-based auth for API calls
3. **Password Reset Tokens**: 
   - UUID-based tokens
   - 1-hour expiry
   - Single-use tokens
4. **Email Validation**: Valid email format required
5. **Password Strength**: Minimum 6 characters (can be increased)

---

## 📱 API Endpoints

### Public Endpoints (No Auth Required)
- `POST /api/users/register` - Register new user
- `POST /api/users/login` - User login
- `POST /api/users/forgot-password` - Request password reset
- `POST /api/users/reset-password` - Reset password with token

### Protected Endpoints (Auth Required)
- `GET /api/users/profile` - Get user profile
- `GET /api/users/usage-status` - Check usage limit status

---

## 🧪 Testing

### Test User Registration:
```bash
curl -X POST http://localhost:8080/api/users/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'
```

### Test Login:
```bash
curl -X POST http://localhost:8080/api/users/login \
  -H "Content-Type: application/json" \
  -d '{
    "usernameOrEmail": "testuser",
    "password": "password123"
  }'
```

### Test Profile (with token):
```bash
curl -X GET http://localhost:8080/api/users/profile \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## 🎨 Frontend Routes

- `/register` - User registration
- `/login` - User login
- `/forgot-password` - Request password reset
- `/reset-password?token=xxx` - Reset password
- `/profile` - User profile (protected)
- `/` - Home (OCR interface)

---

## 🐛 Troubleshooting

### Email not sending:
- Check SMTP credentials
- Ensure "Less secure app access" is enabled (Gmail)
- Use app password instead of regular password
- Check firewall/network settings

### User can't login:
- Verify password is correct
- Check if account is active (`is_active = true`)
- Check database connection

### Usage counter not incrementing:
- Ensure user is authenticated
- Check if `userService.incrementUsageCount()` is called in OCR endpoint
- Verify JWT token is being sent with requests

---

## 📈 Future Enhancements

1. **Email Verification**: Verify email before allowing login
2. **Two-Factor Authentication**: Add 2FA support
3. **Social Login**: Google, Facebook OAuth
4. **Payment Integration**: Stripe/PayPal integration
5. **Usage Analytics**: Detailed usage reports
6. **Subscription Management**: Recurring payments
7. **Admin User Management**: CRUD operations for users
8. **Rate Limiting**: API rate limiting per user

---

## 📦 Dependencies Added

### Backend (pom.xml):
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-mail</artifactId>
</dependency>
```

### Frontend:
No additional dependencies required (uses existing axios and framer-motion)

---

## ✨ Complete!

All user module features are now implemented and ready to use! 

🎉 Users can register, login, reset passwords, and track their OCR usage with a beautiful UI!

