# Lipika - Ranjana Script OCR System

<div align="center">

![Lipika Logo](frontend/src/images/Logo.png)

**Advanced Optical Character Recognition System for Ranjana Script**

</div>

---

## About The Project

**Lipika** is a comprehensive web-based Optical Character Recognition (OCR) system specifically designed to recognize and digitize **Ranjana script** - an ancient Brahmic script historically used in Nepal and parts of India for writing Sanskrit, Newari, and Tibetan languages.

The system combines modern deep learning techniques with a user-friendly web interface to preserve and digitize historical manuscripts, religious texts, and cultural documents written in Ranjana script.

### Key Objectives

- **Preserve Cultural Heritage**: Digitize ancient Ranjana script documents for future generations
- **Accessibility**: Make historical texts accessible to researchers and scholars worldwide
- **Accuracy**: Provide high-accuracy character recognition using deep learning
- **User-Friendly**: Simple interface for both technical and non-technical users

---

## Features

### User Management
- **User Registration & Authentication** with JWT-based security
- **Role-Based Access Control** (User & Admin roles)
- **User Profile Management** with avatar upload
- **Password Reset** via email OTP verification
- **Free & Premium Subscriptions** with daily usage limits

### OCR Capabilities
- **Image Upload OCR**: Upload images directly for recognition
- **Camera Capture OCR**: Real-time capture using device camera
- **Batch Processing**: Process multiple images efficiently
- **High Accuracy**: Deep learning model trained on 60+ Ranjana characters
- **Format Support**: JPEG, PNG, GIF, WebP

### Translation Services
- **Multi-Language Translation**: Translate recognized text to:
  - Nepali (Devanagari)
  - English (Romanized)
- **Real-time Translation**: Instant translation results
- **Translation History**: Track all translations

### Analytics & Reporting
- **User Dashboard**: Track personal OCR usage and statistics
- **Admin Analytics**: System-wide usage metrics and insights
- **Character Recognition Stats**: Detailed character-level statistics
- **Usage History**: Complete OCR history with timestamps

### Subscription Management
- **Free Tier**: 10 OCR scans per day
- **Premium Monthly**: Unlimited scans (NPR 100/month)
- **Premium Yearly**: Unlimited scans with 17% savings (NPR 1000/year)
- **Flexible Extensions**: Easy subscription renewal for premium users

### Admin Panel
- **User Management**: View, activate, deactivate users
- **OCR History Monitoring**: Track all system OCR operations
- **Contact Form Management**: Handle user inquiries
- **System Analytics**: Comprehensive dashboard with charts
- **Character Statistics**: Monitor recognition accuracy per character

### Communication
- **Email Notifications**: Welcome emails, password reset, confirmations
- **Contact Form**: Direct communication channel with support
- **SMTP Integration**: Reliable email delivery

### User Experience
- **Modern UI/UX**: Clean, intuitive interface with Framer Motion animations
- **Responsive Design**: Seamless experience across desktop, tablet, and mobile
- **Dark Mode Ready**: Eye-friendly interface
- **Real-time Feedback**: Loading states, progress indicators, success/error messages

---

## Tech Stack

### Frontend
- **Framework**: React 18.2 with React Router DOM
- **Build Tool**: Vite 5.2
- **Styling**: Tailwind CSS 3.4
- **Animations**: Framer Motion 11.0
- **Icons**: React Icons 5.5
- **HTTP Client**: Axios 1.6
- **Camera**: React Webcam 7.2
- **Charts**: Recharts 3.4
- **Security**: Crypto-JS for encryption

### Backend (Java Spring Boot)
- **Framework**: Spring Boot 3.2.0
- **Language**: Java 17
- **Security**: Spring Security + JWT Authentication
- **Database**: MySQL with Spring Data JPA
- **Email**: Spring Boot Mail (SMTP)
- **Password Hashing**: BCrypt
- **HTTP Client**: Spring WebFlux (for Python service communication)
- **Build Tool**: Maven
- **API Documentation**: Spring Actuator (health checks)

### Machine Learning (Python)
- **Framework**: PyTorch 2.0+
- **Computer Vision**: OpenCV, Pillow
- **OCR Engine**: Custom CRNN model + EasyOCR
- **Web Service**: Flask 2.0 with Flask-CORS
- **Data Processing**: NumPy, Matplotlib
- **Training Tools**: TorchVision, TQDM

### Database
- **Primary Database**: MySQL 8.0+
- **ORM**: Hibernate (via Spring Data JPA)
- **Connection Pooling**: HikariCP

### Development Tools
- **Version Control**: Git & GitHub
- **IDE**: Visual Studio Code, Eclipse
- **API Testing**: Postman/Thunder Client
- **Package Management**: npm (frontend), Maven (backend), pip (Python)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Auth Pages   │  │ OCR Pages    │  │ Admin Panel  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Backend (Spring Boot)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Controllers  │  │ Services     │  │ Security     │          │
│  │              │  │              │  │ (JWT + Auth) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
│         │                  │                                      │
│         ▼                  ▼                                      │
│  ┌─────────────────────────────────┐                            │
│  │   Spring Data JPA (Hibernate)   │                            │
│  └─────────────┬───────────────────┘                            │
└────────────────┼──────────────────────────────────────────┬─────┘
                 │                                           │
                 ▼                                           │ HTTP
        ┌─────────────────┐                                 ▼
        │  MySQL Database │                    ┌──────────────────────┐
        │                 │                    │  Python OCR Service  │
        │  - Users        │                    │  (Flask + PyTorch)   │
        │  - OCR History  │                    │                      │
        │  - Contacts     │                    │  ┌────────────────┐ │
        │  - OTPs         │                    │  │ CRNN Model     │ │
        └─────────────────┘                    │  │ (Ranjana OCR)  │ │
                                               │  └────────────────┘ │
                                               └──────────────────────┘
```

---

## Prerequisites

Before you begin, ensure you have the following installed:

### Required Software
- **Node.js** 18+ and npm 9+
- **Java** 17 or higher (JDK)
- **Maven** 3.8+
- **MySQL** 8.0+
- **Python** 3.8+
- **Git**

### Recommended Tools
- **Eclipse IDE** or **IntelliJ IDEA** (for Java backend)
- **Visual Studio Code** (for frontend)
- **MySQL Workbench** or **DBeaver** (database management)
- **Postman** (API testing)

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/lipika-ocr.git
cd lipika-ocr
```

### 2. Database Setup
```sql
-- Create database
CREATE DATABASE lipika;

-- Create admin user (optional)
USE lipika;

-- Tables will be auto-created by Spring Boot JPA
```

### 3. Backend Setup (Spring Boot)

#### Configure Database
Edit `javabackend/src/main/resources/application.properties`:
```properties
spring.datasource.url=jdbc:mysql://localhost:3306/lipika
spring.datasource.username=your_mysql_username
spring.datasource.password=your_mysql_password
```

#### Configure Email (SMTP)
Create `javabackend/src/main/resources/application-mail.properties`:
```properties
spring.mail.username=your-email@gmail.com
spring.mail.password=your-app-password
```
> **Note**: Use Gmail App Password, not your regular password

#### Build and Run
```bash
cd javabackend
mvn clean install
mvn spring-boot:run
```
Backend will start on `http://localhost:8080`

### 4. Frontend Setup (React)
```bash
cd frontend
npm install
npm run dev
```
Frontend will start on `http://localhost:5173`

### 5. Python OCR Service Setup
```bash
cd python-model
pip install -r requirements.txt
python app.py
```
Python service will start on `http://localhost:5000`

---

## Usage

### For End Users

1. **Register/Login**
   - Visit `http://localhost:5173`
   - Create an account or login
   - Free tier: 10 OCR scans per day

2. **Upload Image**
   - Navigate to OCR page
   - Upload image or use camera
   - View recognized text and translation

3. **Upgrade to Premium**
   - Visit Pricing page
   - Choose monthly or yearly plan
   - Enjoy unlimited OCR scans

### For Administrators

1. **Login to Admin Panel**
   - Visit `http://localhost:5173/admin`
   - Login with admin credentials
   - Access admin dashboard

2. **Manage Users**
   - View all registered users
   - Activate/deactivate accounts
   - Monitor user activity

3. **View Analytics**
   - System-wide OCR statistics
   - Character recognition accuracy
   - User engagement metrics

---

## Project Structure

```
lipika-ocr/
├── frontend/                      # React Frontend
│   ├── src/
│   │   ├── components/           # Reusable components
│   │   ├── pages/                # Page components
│   │   │   ├── admin/           # Admin panel pages
│   │   │   ├── About.jsx
│   │   │   ├── Contact.jsx
│   │   │   ├── ForgotPassword.jsx
│   │   │   ├── Home.jsx
│   │   │   ├── Login.jsx
│   │   │   ├── OCR.jsx
│   │   │   ├── Payment.jsx
│   │   │   ├── Pricing.jsx
│   │   │   └── UserProfile.jsx
│   │   ├── context/             # React Context (Auth)
│   │   ├── services/            # API services
│   │   ├── config/              # Configuration
│   │   ├── images/              # Static images
│   │   └── App.jsx              # Main app component
│   ├── package.json
│   └── vite.config.js
│
├── javabackend/                  # Spring Boot Backend
│   ├── src/main/java/com/lipika/
│   │   ├── controller/          # REST Controllers
│   │   ├── service/             # Business Logic
│   │   ├── repository/          # Data Access Layer
│   │   ├── model/               # Entity Models
│   │   ├── dto/                 # Data Transfer Objects
│   │   ├── security/            # Security Config
│   │   ├── util/                # Utilities
│   │   └── LipikaApplication.java
│   ├── src/main/resources/
│   │   ├── application.properties
│   │   └── application-mail.properties.example
│   └── pom.xml
│
├── python-model/                 # Python OCR Service
│   ├── app.py                   # Flask app
│   ├── model.py                 # PyTorch CRNN model
│   ├── train.py                 # Training script
│   ├── inference.py             # Inference script
│   ├── requirements.txt
│   └── best_model.pth           # Trained model
│
├── Dataset/                      # Training dataset
│   ├── a/, aa/, ah/, ai/, ...  # Character folders
│   └── ...
│
└── README.md                     # This file
```

---

## API Endpoints

### Authentication
- `POST /api/users/register` - Register new user
- `POST /api/users/login` - User login
- `POST /api/admin/auth/login` - Admin login
- `POST /api/auth/forgot-password/request` - Request OTP
- `POST /api/auth/forgot-password/verify` - Verify OTP
- `POST /api/auth/forgot-password/reset` - Reset password

### OCR Operations
- `POST /api/ocr/process` - Process image for OCR
- `GET /api/ocr/history` - Get OCR history
- `GET /api/images` - Retrieve uploaded images

### Translation
- `POST /api/translate` - Translate text

### Admin
- `GET /api/admin/users` - Get all users
- `PUT /api/admin/users/{id}/activate` - Activate user
- `DELETE /api/admin/users/{id}` - Delete user
- `GET /api/admin/ocr-history` - All OCR history
- `GET /api/admin/analytics` - System analytics
- `GET /api/admin/contacts` - Contact messages
- `PUT /api/admin/contacts/{id}/read` - Mark contact as read

### User Profile
- `GET /api/users/profile` - Get user profile
- `PUT /api/users/profile` - Update profile
- `POST /api/users/profile/avatar` - Upload avatar

### Contact
- `POST /api/contact/submit` - Submit contact form

---

## Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: BCrypt for secure password storage
- **Role-Based Access**: User and Admin role separation
- **CORS Protection**: Configured CORS policies
- **SQL Injection Prevention**: Parameterized queries via JPA
- **XSS Protection**: Input sanitization
- **CSRF Protection**: Spring Security CSRF tokens

---

## UI Screenshots

*(Add screenshots here when available)*

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Authors

- **Your Name** - *Initial work* - [GitHub](https://github.com/yourusername)

---

## Acknowledgments

- Ranjana script historical research and documentation
- PyTorch community for deep learning resources
- Spring Boot and React communities
- All contributors and testers

---

## Support

For support, email sajesh.bajracharya.01@gmail.com or create an issue in the repository.

---

<div align="center">

**Made with love for preserving cultural heritage**

</div>
