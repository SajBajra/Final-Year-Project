# Lipika Backend - Spring Boot MVC

Spring Boot backend implementing the **Presenter Layer** of the Lipika OCR system, following MVC (Model-View-Controller) pattern.

## Architecture

```
javabackend/
├── src/main/java/com/lipika/
│   ├── config/              # Configuration classes
│   │   ├── WebConfig.java   # CORS configuration
│   │   └── ApplicationConfig.java  # Bean configuration
│   ├── controller/          # REST Controllers (View Layer in MVC)
│   │   ├── OCRController.java
│   │   ├── TranslationController.java
│   │   └── HealthController.java
│   ├── service/             # Business Logic (Model Layer in MVC)
│   │   ├── OCRService.java
│   │   ├── TranslationService.java
│   │   └── impl/
│   │       ├── OCRServiceImpl.java
│   │       └── TranslationServiceImpl.java
│   ├── model/               # Data Transfer Objects (DTOs)
│   │   ├── OCRRequest.java
│   │   ├── OCRResponse.java
│   │   ├── TranslationRequest.java
│   │   ├── TranslationResponse.java
│   │   └── ApiResponse.java
│   ├── exception/           # Exception handling
│   │   └── GlobalExceptionHandler.java
│   └── LipikaApplication.java  # Main application class
└── src/main/resources/
    ├── application.properties
    └── application.yml
```

## MVC Pattern Implementation

### **Model Layer** (Service + DTOs)
- **Services**: Business logic and integration with Python OCR service
  - `OCRService`: Handles OCR recognition by calling Python service
  - `TranslationService`: Translates Ranjana text to English
- **DTOs**: Data transfer objects for API requests/responses
  - `OCRRequest`, `OCRResponse`
  - `TranslationRequest`, `TranslationResponse`
  - `ApiResponse`: Generic wrapper for all API responses

### **View Layer** (REST Controllers)
- **Controllers**: REST API endpoints
  - `OCRController`: `/api/ocr/*` endpoints
  - `TranslationController`: `/api/translate/*` endpoints
  - `HealthController`: `/api/health` endpoint

### **Controller Layer** (Configuration)
- **Config**: Application configuration
  - `WebConfig`: CORS settings
  - `ApplicationConfig`: Bean definitions (RestTemplate, WebClient)
- **Exception Handler**: Global error handling

## REST API Endpoints

### OCR Endpoints

#### POST `/api/ocr/recognize`
Recognize text from uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "success": true,
  "message": "Text recognized successfully",
  "data": {
    "success": true,
    "text": "नेपाली भाषा",
    "characters": [
      {
        "character": "न",
        "confidence": 0.95,
        "bbox": {
          "x": 10,
          "y": 5,
          "width": 25,
          "height": 30
        },
        "index": 0
      }
    ],
    "confidence": 95.5,
    "count": 6
  }
}
```

#### GET `/api/ocr/health`
Health check for OCR service.

---

### Translation Endpoints

#### POST `/api/translate`
Translate Ranjana text to target language.

**Request:**
```json
{
  "text": "नेपाली भाषा",
  "targetLanguage": "en"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Translation successful",
  "data": {
    "originalText": "नेपाली भाषा",
    "translatedText": "Nepali language",
    "sourceLanguage": "ranjana",
    "targetLanguage": "en",
    "success": true
  }
}
```

#### POST `/api/translate/text`
Quick text translation (query parameters).

**Request:**
- URL: `/api/translate/text?text=नेपाली&targetLanguage=en`

**Response:**
```json
{
  "success": true,
  "data": "Nepali"
}
```

---

### Health Endpoint

#### GET `/api/health`
Service health check.

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "UP",
    "timestamp": "2025-01-15T10:30:00",
    "service": "Lipika Backend - Presenter Layer",
    "version": "1.0.0"
  }
}
```

## Setup & Running

### Prerequisites
- Java 17+
- Maven 3.6+

### Build & Run

```bash
# Navigate to backend directory
cd javabackend

# Build project
mvn clean install

# Run application
mvn spring-boot:run
```

The service will start on `http://localhost:8080`

### Configuration

Edit `application.properties` or `application.yml`:

```properties
# OCR Service URL (Python service)
ocr.service.url=http://localhost:5000

# Server Port
server.port=8080

# File Upload Size Limit
spring.servlet.multipart.max-file-size=10MB
```

## Integration with Python OCR Service

The backend calls the Python OCR service at `http://localhost:5000`:

1. Frontend → Java Backend (`/api/ocr/recognize`)
2. Java Backend → Python OCR Service (`/predict`)
3. Python OCR Service → Java Backend (OCR results)
4. Java Backend → Frontend (formatted response)

## Features

✅ **MVC Pattern**: Clean separation of concerns  
✅ **REST APIs**: RESTful endpoints for all operations  
✅ **Error Handling**: Global exception handler  
✅ **Validation**: Request validation using Bean Validation  
✅ **CORS**: Configured for React frontend  
✅ **Logging**: Comprehensive logging for debugging  
✅ **Health Checks**: Service health monitoring  

## Testing

```bash
# Test health endpoint
curl http://localhost:8080/api/health

# Test OCR endpoint
curl -X POST http://localhost:8080/api/ocr/recognize \
  -F "image=@test_image.png"

# Test translation endpoint
curl -X POST http://localhost:8080/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"नेपाली","targetLanguage":"en"}'
```

## Next Steps

- [ ] Add database support for storing OCR history
- [ ] Integrate with Google Translate API for better translations
- [ ] Add authentication and authorization
- [ ] Add rate limiting
- [ ] Add caching for translations
- [ ] Add metrics and monitoring

---

**Note**: Make sure the Python OCR service is running on port 5000 before using OCR endpoints.

## Troubleshooting

### Error: "Could not find or load main class com.lipika.LipikaApplication"

This error occurs when the project hasn't been compiled yet. Follow these steps:

**Solution 1: Build with Maven (Recommended)**
```bash
cd javabackend

# Clean and build the project
mvn clean install

# Run the application
mvn spring-boot:run
```

**Solution 2: Build JAR and Run**
```bash
cd javabackend

# Build the JAR file
mvn clean package

# Run the JAR file
java -jar target/lipika-backend-1.0.0.jar
```

**Solution 3: If Maven is not installed**

1. **Install Maven:**
   - Download from: https://maven.apache.org/download.cgi
   - Or use package manager:
     - Windows (Chocolatey): `choco install maven`
     - Windows (Winget): `winget install Apache.Maven`

2. **Verify Maven installation:**
   ```bash
   mvn --version
   ```

3. **Then build the project:**
   ```bash
   cd javabackend
   mvn clean install
   mvn spring-boot:run
   ```

**Solution 4: Use IDE (IntelliJ IDEA / Eclipse)**

1. Open the `javabackend` folder in your IDE
2. Let the IDE auto-import Maven dependencies
3. Right-click on `LipikaApplication.java`
4. Select "Run 'LipikaApplication'"

**Common Issues:**

- **Java version mismatch**: Ensure you have Java 17 installed (`java -version`)
- **Maven not found**: Install Maven and add it to your PATH
- **Compilation errors**: Check that all dependencies are downloaded correctly

