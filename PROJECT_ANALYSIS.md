# 📊 Lipika OCR System - Comprehensive Project Analysis

## 🎯 Project Overview

**Lipika** is a full-stack OCR (Optical Character Recognition) system designed specifically for **Ranjana Script** recognition, converting it to **Devanagari** text. The system features a three-tier architecture with AR (Augmented Reality) capabilities, user authentication, trial tracking, and a comprehensive admin panel.

---

## 🏗️ Architecture Overview

### **Three-Tier Architecture (MVP Pattern)**

1. **View Layer (Frontend)** - React.js
2. **Presenter Layer (Backend)** - Spring Boot (Java)
3. **Model Layer (OCR Engine)** - Python (PyTorch/Flask)

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                      │
│  - User Interface (Home, Features, About, Login)        │
│  - Admin Panel (Dashboard, Analytics, History, Users)   │
│  - OCR Components (Image Upload, Camera, AR Overlay)    │
│  - Authentication & Authorization                        │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/REST API
┌──────────────────▼──────────────────────────────────────┐
│              Backend (Spring Boot)                       │
│  - REST API Endpoints                                    │
│  - Authentication & JWT                                  │
│  - User Management & Trial Tracking                      │
│  - OCR History & Analytics                               │
│  - Database Management (MySQL)                           │
└──────────────────┬──────────────────────────────────────┘
                   │ HTTP/REST API
┌──────────────────▼──────────────────────────────────────┐
│            OCR Service (Python/Flask)                   │
│  - CRNN Model (PyTorch)                                  │
│  - Character Segmentation (OpenCV)                        │
│  - Text Recognition & Post-processing                      │
│  - AR-Ready Bounding Boxes                               │
└──────────────────────────────────────────────────────────┘
```

---

## 📦 Technology Stack

### **Frontend**
- **Framework**: React 18.2.0
- **Build Tool**: Vite 5.2.0
- **Routing**: React Router DOM 6.30.1
- **Styling**: Tailwind CSS 3.4.3
- **Animations**: Framer Motion 11.0.25
- **Icons**: React Icons 5.5.0
- **Charts**: Recharts 3.4.1
- **HTTP Client**: Axios 1.6.8
- **Webcam**: React Webcam 7.2.0

### **Backend**
- **Framework**: Spring Boot 3.2.0
- **Language**: Java 17
- **Security**: Spring Security + JWT (jjwt 0.12.3)
- **Database**: MySQL 8.0 (via XAMPP)
- **ORM**: Spring Data JPA / Hibernate
- **Password Hashing**: BCrypt
- **HTTP Client**: Spring WebFlux (for Python service calls)
- **Build Tool**: Maven
- **Utilities**: Lombok

### **OCR Service**
- **Framework**: Flask 2.0.0
- **Deep Learning**: PyTorch 2.0.0+, Torchvision 0.15.0+
- **Image Processing**: OpenCV 4.5.0+, Pillow 9.0.0+
- **Utilities**: NumPy, Matplotlib, tqdm
- **Optional**: EasyOCR 1.7.0

### **Database**
- **RDBMS**: MySQL 8.0
- **Database Name**: `lipika`
- **Character Set**: utf8mb4 (for Devanagari support)

---

## 📁 Project Structure

```
FYP/
├── frontend/                    # React Frontend Application
│   ├── src/
│   │   ├── components/          # Reusable React components
│   │   │   ├── Header.jsx      # Navigation bar with auth
│   │   │   ├── Footer.jsx      # Footer component
│   │   │   ├── ImageUpload.jsx # Image upload component
│   │   │   ├── CameraCapture.jsx # Webcam capture
│   │   │   ├── OCRResult.jsx   # OCR results display
│   │   │   ├── AROverlay.jsx   # AR overlay component
│   │   │   ├── TrialCounter.jsx # Trial limit display
│   │   │   ├── AdminLayout.jsx # Admin panel layout
│   │   │   ├── ProtectedRoute.jsx # Route guards
│   │   │   └── ConfirmModal.jsx # Confirmation dialogs
│   │   ├── pages/              # Page components
│   │   │   ├── Home.jsx        # Main OCR page
│   │   │   ├── Features.jsx    # Features page
│   │   │   ├── About.jsx       # About page
│   │   │   ├── Login.jsx       # Login page
│   │   │   ├── Register.jsx    # Registration page
│   │   │   └── admin/          # Admin panel pages
│   │   │       ├── AdminDashboard.jsx
│   │   │       ├── AdminOCRHistory.jsx
│   │   │       ├── AdminAnalytics.jsx
│   │   │       ├── AdminCharacterStats.jsx
│   │   │       ├── AdminUserManagement.jsx
│   │   │       └── AdminSettings.jsx
│   │   ├── services/           # API service functions
│   │   │   ├── ocrService.js   # OCR API calls
│   │   │   └── adminService.js # Admin API calls
│   │   ├── context/            # React Context
│   │   │   └── AuthContext.jsx # Authentication state
│   │   ├── config/             # Configuration
│   │   │   └── constants.js    # Routes, API endpoints
│   │   └── utils/              # Utility functions
│   │       └── cookieUtils.js  # Cookie management
│   └── package.json
│
├── javabackend/                 # Spring Boot Backend
│   ├── src/main/java/com/lipika/
│   │   ├── config/             # Configuration classes
│   │   │   ├── SecurityConfig.java      # Spring Security
│   │   │   ├── WebConfig.java           # CORS, Web config
│   │   │   ├── DataInitializer.java     # Default admin user
│   │   │   └── ApplicationConfig.java    # App config
│   │   ├── controller/         # REST Controllers
│   │   │   ├── OCRController.java       # OCR endpoints
│   │   │   ├── AuthController.java      # Auth endpoints
│   │   │   ├── AdminController.java     # Admin endpoints
│   │   │   ├── TranslationController.java # Translation
│   │   │   ├── UserManagementController.java # User mgmt
│   │   │   └── HealthController.java     # Health check
│   │   ├── service/            # Business logic
│   │   │   ├── OCRService.java
│   │   │   ├── AuthService.java
│   │   │   ├── AdminService.java
│   │   │   ├── TranslationService.java
│   │   │   ├── TrialTrackingService.java
│   │   │   └── impl/           # Service implementations
│   │   ├── repository/         # Data access layer
│   │   │   ├── OCRHistoryRepository.java
│   │   │   ├── UserRepository.java
│   │   │   └── TrialTrackingRepository.java
│   │   ├── model/              # Entity classes
│   │   │   ├── User.java
│   │   │   ├── OCRHistory.java
│   │   │   ├── TrialTracking.java
│   │   │   └── dto/            # Data Transfer Objects
│   │   ├── security/           # Security components
│   │   │   ├── JwtAuthenticationFilter.java
│   │   │   └── JwtUtil.java
│   │   └── exception/          # Exception handling
│   │       └── GlobalExceptionHandler.java
│   ├── database/
│   │   ├── schema.sql          # Database schema
│   │   └── README.md           # Database docs
│   └── pom.xml
│
├── python-model/                # Python OCR Service
│   ├── ocr_service_ar.py        # Main OCR service (AR-ready)
│   ├── ocr_service.py           # Basic OCR service
│   ├── train_character_crnn_improved.py # Model training
│   ├── prepare_combined_dataset.py      # Dataset prep
│   ├── convert_labels_to_ranjana.py     # Label conversion
│   ├── best_character_crnn_improved.pth  # Trained model
│   ├── chars.txt                # Character set
│   ├── checkpoints/             # Training checkpoints
│   └── requirements.txt
│
├── Dataset/                     # Training dataset
│   └── [character_folders]/    # One folder per character
│
└── prepared_dataset/            # Processed dataset
    ├── images/                  # Processed images
    ├── train_labels.txt         # Training labels
    └── val_labels.txt           # Validation labels
```

---

## 🔑 Key Features

### **1. OCR Functionality**
- **Character-Based Recognition**: Recognizes individual Ranjana script characters
- **Segmentation-Based**: Uses OpenCV contour detection for character isolation
- **AR-Ready**: Provides bounding boxes for each character for AR overlay
- **Multi-Input Support**: Image upload, webcam capture
- **Real-Time Processing**: Fast inference using PyTorch
- **Post-Processing**: Text cleanup and formatting

### **2. User Authentication & Authorization**
- **User Registration**: Username, email, password
- **JWT-Based Authentication**: Stateless token-based auth
- **Role-Based Access Control**: USER and ADMIN roles
- **Password Security**: BCrypt hashing
- **Default Admin User**: Auto-created on startup
  - Username: `admin`
  - Password: `admin123`
  - Email: `admin@lipika.com`

### **3. Trial System**
- **10 Free OCR Attempts** for unregistered users
- **Multi-Factor Tracking**:
  - IP Address tracking
  - Cookie-based tracking (`lipika_trial_id`)
  - Browser fingerprinting (User-Agent + headers)
- **Bypass Prevention**: IP tracking prevents incognito bypass
- **Trial Counter**: Real-time display of remaining trials

### **4. Admin Panel**
- **Dashboard**: 
  - Total OCR records
  - Text length distribution (pie chart)
  - Recent activity statistics
- **OCR History**:
  - Search by recognized text
  - Filter by date range
  - Sort by timestamp, character count
  - Pagination (10 records per page)
  - Bulk delete with confirmation modal
  - CSV export
- **Analytics**:
  - Time-series charts (daily/weekly/monthly)
  - Text length distribution
  - Usage trends
- **Character Statistics**:
  - Top 20 most recognized characters
  - Character frequency analysis
  - Bar charts for visualization
- **User Management**:
  - View all users
  - Filter by role and status
  - Activate/deactivate users
  - View user OCR history
- **Settings**:
  - Change admin password
  - System configuration

### **5. Translation Feature**
- **Optional Translation**: Devanagari → English
- **External API**: LibreTranslate (https://libretranslate.de)
- **User-Triggered**: Only translates when user clicks "Translate to English"

### **6. AR Overlay**
- **Bounding Boxes**: Character-level bounding boxes
- **Real-Time Overlay**: Overlay recognized text on original image
- **Toggle View**: Switch between normal and AR view

---

## 🗄️ Database Schema

### **Tables**

1. **`users`**
   - `id` (BIGINT, PK)
   - `username` (VARCHAR(100), UNIQUE)
   - `email` (VARCHAR(255), UNIQUE)
   - `password_hash` (VARCHAR(255))
   - `role` (VARCHAR(20)) - 'USER' or 'ADMIN'
   - `is_active` (BOOLEAN)
   - `created_at`, `updated_at`, `last_login` (DATETIME)

2. **`ocr_history`**
   - `id` (BIGINT, PK)
   - `user_id` (BIGINT, FK → users.id, NULLABLE)
   - `is_registered` (BOOLEAN)
   - `ip_address` (VARCHAR(45))
   - `cookie_id` (VARCHAR(255))
   - `image_filename` (VARCHAR(500))
   - `recognized_text` (TEXT)
   - `character_count` (INT)
   - `confidence` (DOUBLE)
   - `timestamp` (DATETIME)
   - `language` (VARCHAR(50)) - Default: 'devanagari'
   - **Indexes**: timestamp, confidence, language, user_id, is_registered, ip_address
   - **Fulltext Index**: recognized_text (for search)

3. **`trial_tracking`**
   - `id` (BIGINT, PK)
   - `ip_address` (VARCHAR(45))
   - `cookie_id` (VARCHAR(255))
   - `fingerprint` (VARCHAR(255))
   - `trial_count` (INT)
   - `first_attempt`, `last_attempt` (DATETIME)
   - `is_blocked` (BOOLEAN)
   - **Unique Key**: (ip_address, cookie_id, fingerprint)

4. **`system_settings`**
   - `id` (BIGINT, PK)
   - `setting_key` (VARCHAR(100), UNIQUE)
   - `setting_value` (TEXT)
   - `description` (VARCHAR(500))
   - `updated_at` (DATETIME)

---

## 🔌 API Endpoints

### **Public Endpoints**
- `GET /api/health` - Health check
- `POST /api/ocr/recognize` - OCR recognition (trial-limited for unregistered)
- `POST /api/translation/translate` - Text translation

### **Authentication Endpoints**
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user (requires auth)

### **Admin Endpoints** (Require ADMIN role)
- `GET /api/admin/dashboard` - Dashboard statistics
- `GET /api/admin/ocr-history` - OCR history with filters
- `DELETE /api/admin/ocr-history/{id}` - Delete single record
- `DELETE /api/admin/ocr-history/bulk` - Bulk delete
- `GET /api/admin/ocr-history/export` - Export to CSV
- `GET /api/admin/analytics` - Analytics data
- `GET /api/admin/characters/stats` - Character statistics
- `GET /api/admin/users` - List users
- `GET /api/admin/users/stats` - User statistics
- `GET /api/admin/users/{userId}/history` - User OCR history
- `PUT /api/admin/users/{userId}/status` - Update user status
- `PUT /api/admin/settings/password` - Change admin password
- `GET /api/admin/diagnostics` - Diagnostic information

---

## 🧠 OCR Model Architecture

### **Model Type**: Character-Based CRNN (Convolutional Recurrent Neural Network)

**Architecture Components:**
1. **CNN Feature Extractor** (5 layers)
   - Residual connections
   - Batch normalization
   - ReLU activations
   - Global average pooling
   - Attention mechanism

2. **Bidirectional LSTM** (3 layers)
   - Hidden size: 256
   - Dropout: 0.3
   - Processes sequences in both directions

3. **Multi-Layer Classifier** (4 layers)
   - Fully connected layers
   - Dropout for regularization
   - Output: 74 character classes

**Input/Output:**
- **Input**: 64x64 grayscale character images
- **Output**: Character class probabilities (74 classes: 66 Devanagari + 8 ASCII)

**Processing Pipeline:**
```
Input Image
    ↓
Preprocessing (grayscale, normalization)
    ↓
Character Segmentation (OpenCV contour detection)
    ↓
Character Isolation (with padding)
    ↓
CRNN Recognition (PyTorch)
    ↓
CTC Decoding / Beam Search
    ↓
Post-processing (text cleanup)
    ↓
Devanagari Text Output
```

**Character Set**: 74 characters
- 66 Devanagari characters (U+0900–U+097F)
- 8 ASCII characters (digits 0-9, some punctuation)

---

## 🔒 Security Features

### **Authentication & Authorization**
- JWT tokens with 24-hour expiration
- Role-based access control (USER, ADMIN)
- Password hashing with BCrypt
- Stateless session management
- Protected routes on frontend

### **Trial System Security**
- Multi-factor tracking (IP + Cookie + Fingerprint)
- Prevents bypass via incognito mode
- Server-side validation

### **CORS Configuration**
- Allowed origins: `http://localhost:5173`, `http://localhost:3000`
- Credentials enabled
- All HTTP methods allowed

### **Security Considerations**
- ⚠️ JWT secret should be changed in production (min 256 bits)
- ⚠️ Password policy: Currently minimum 6 characters
- ⚠️ Rate limiting: Not implemented (consider adding)
- ⚠️ HTTPS: Required in production
- ⚠️ Cookie security: Set secure and httpOnly flags in production

---

## 📊 Current Status

### **✅ Completed Features**
1. ✅ Full-stack OCR system with three-tier architecture
2. ✅ User authentication and authorization
3. ✅ Trial tracking system (10 free attempts)
4. ✅ Admin panel with comprehensive features
5. ✅ OCR history tracking and analytics
6. ✅ Character statistics and visualization
7. ✅ User management (admin)
8. ✅ Translation feature (optional)
9. ✅ AR overlay with bounding boxes
10. ✅ Responsive design (mobile + desktop)
11. ✅ Default admin user auto-creation
12. ✅ CSV export functionality
13. ✅ Confirmation modals for destructive actions
14. ✅ Icons on all admin page headings
15. ✅ Centered settings page layout

### **⚠️ Known Issues**
1. **Python 3.13 Compatibility**: Potential issues with PyTorch/torchvision (see `TROUBLESHOOT_PYTHON313.md`)
2. **Accidental File**: `tatus` file exists in root (should be cleaned up)
3. **Uncommitted Changes**: `AdminController.java` has uncommitted modifications

### **📝 Pending Tasks**
1. Clean up `tatus` file
2. Commit pending changes in `AdminController.java`
3. Consider adding rate limiting for OCR endpoints
4. Strengthen password policy
5. Add HTTPS support for production
6. Implement cookie security flags for production

---

## 🚀 Deployment & Running

### **Prerequisites**
- Node.js 18+ (for frontend)
- Java 17+ (for backend)
- Python 3.8-3.12 (for OCR service, avoid 3.13)
- MySQL 8.0 (via XAMPP)
- Maven (for backend build)

### **Running the System**

**1. Start MySQL (XAMPP)**
- Start MySQL service in XAMPP Control Panel
- Database `lipika` should be created automatically

**2. Start Python OCR Service**
```powershell
cd python-model
python ocr_service_ar.py
```
- Runs on `http://localhost:5000`

**3. Start Java Backend**
```powershell
cd javabackend
mvn spring-boot:run
```
- Runs on `http://localhost:8080`
- Auto-creates default admin user on first startup

**4. Start React Frontend**
```powershell
cd frontend
npm install  # First time only
npm run dev
```
- Runs on `http://localhost:5173` (Vite default) or `http://localhost:3000`

### **Default Credentials**
- **Admin Username**: `admin`
- **Admin Password**: `admin123`
- **Admin Email**: `admin@lipika.com`

---

## 📈 Performance Metrics

### **Model Performance**
- **Validation Accuracy**: ~99.06%
- **Training Accuracy**: ~98.93%
- **Model Type**: ImprovedCharacterCRNN
- **Training Epochs**: 138 (from checkpoint)

### **Dataset Statistics**
- **Total Character Classes**: 62-74 characters
- **Total Images**: ~13,584 images
- **Average Images per Character**: ~219 images
- **Character Set**: Ranjana → Devanagari

### **System Performance**
- **OCR Processing**: Real-time (depends on image size)
- **Database**: Indexed for fast queries
- **Frontend**: Optimized with React and Vite
- **API Response Time**: < 500ms (typical)

---

## 🎯 Use Cases

1. **Historical Document Digitization**: Convert Ranjana script documents to Devanagari
2. **Educational Tools**: Learn Ranjana script with AR overlay
3. **Cultural Preservation**: Preserve Ranjana script in digital format
4. **Research**: Analyze historical texts in Ranjana script
5. **Accessibility**: Make Ranjana script accessible via OCR

---

## 🔮 Future Enhancements

### **Potential Improvements**
1. **Model Training**:
   - Increase dataset size (target: 500-1000 images per character)
   - Data augmentation improvements
   - Fine-tuning for better accuracy

2. **Features**:
   - Batch OCR processing
   - PDF support
   - Multi-language support
   - Handwriting recognition
   - Real-time video OCR

3. **Infrastructure**:
   - Docker containerization
   - Kubernetes deployment
   - Load balancing
   - Caching layer (Redis)
   - Message queue for async processing

4. **Security**:
   - Rate limiting
   - OAuth2 integration
   - Two-factor authentication
   - Audit logging

5. **User Experience**:
   - Dark mode
   - Internationalization (i18n)
   - Progressive Web App (PWA)
   - Offline support

---

## 📚 Documentation Files

- `ADMIN_FEATURES_IMPLEMENTED.md` - Admin features documentation
- `AUTHENTICATION_IMPLEMENTATION.md` - Auth system docs
- `OCR_SYSTEM_TYPE.md` - OCR system classification
- `TRAINING_ANALYSIS.md` - Training analysis
- `TRAINING_RECOMMENDATIONS.md` - Training recommendations
- `NEXT_STEPS.md` - Next steps guide
- `TROUBLESHOOT_PYTHON313.md` - Python 3.13 troubleshooting
- `javabackend/database/README.md` - Database documentation

---

## 🏆 Project Highlights

1. **Full-Stack Implementation**: Complete three-tier architecture
2. **Modern Tech Stack**: React, Spring Boot, PyTorch
3. **Production-Ready Features**: Authentication, authorization, analytics
4. **AR Capabilities**: Bounding boxes for AR overlay
5. **Comprehensive Admin Panel**: Dashboard, analytics, user management
6. **Trial System**: Smart tracking with multi-factor identification
7. **Responsive Design**: Works on mobile and desktop
8. **Well-Documented**: Extensive documentation and comments

---

## 📞 Support & Maintenance

### **Configuration Files**
- `frontend/vite.config.js` - Frontend build config
- `javabackend/src/main/resources/application.properties` - Backend config
- `python-model/requirements.txt` - Python dependencies

### **Key Configuration Values**
- **JWT Secret**: `lipika-secret-key-change-this-in-production-...`
- **JWT Expiration**: 86400000ms (24 hours)
- **OCR Service URL**: `http://localhost:5000`
- **Translation API**: `https://libretranslate.de/translate`
- **Trial Limit**: 10 attempts
- **Database**: MySQL on `localhost:3306`

---

**Last Updated**: 2024
**Project Status**: ✅ Production-Ready (with minor improvements needed)
**Version**: 1.0.0

