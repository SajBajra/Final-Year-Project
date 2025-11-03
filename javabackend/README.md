# Java Backend - Presenter Layer

## Overview
Spring Boot backend that acts as the presenter layer, coordinating between the React frontend and Python OCR service.

## Responsibilities
- Receive image uploads from frontend
- Call Python OCR API service
- Handle business logic and validation
- Format responses for frontend
- Integrate translation APIs (if needed)
- Manage user sessions and data persistence

## Technology Stack
- Java 17+
- Spring Boot 3.x
- Spring Web (REST API)
- Maven or Gradle

## Setup

### Prerequisites
- Java 17 or higher
- Maven 3.8+ or Gradle 7+

### Build & Run
```bash
# Maven
mvn clean install
mvn spring-boot:run

# Gradle
./gradlew build
./gradlew bootRun
```

## API Integration

Example code to call Python OCR service:
```java
@RestController
public class OCRController {
    
    private final String OCR_SERVICE_URL = "http://localhost:5000";
    
    @PostMapping("/api/ocr")
    public ResponseEntity<OCRResponse> recognizeText(
        @RequestParam("image") MultipartFile image
    ) {
        // Call Python OCR service
        // Return formatted response to frontend
    }
}
```

## Project Structure (To Be Created)
```
javabackend/
├── src/
│   └── main/
│       ├── java/
│       │   └── com/
│       │       └── ranjanapyr/
│       │           ├── controller/   # REST controllers
│       │           ├── service/      # Business logic
│       │           ├── model/        # Data models
│       │           └── config/       # Configuration
│       └── resources/
│           └── application.properties
├── pom.xml                          # Maven dependencies
└── README.md
```

## Next Steps
1. Create Spring Boot project structure
2. Implement OCR controller
3. Add translation integration
4. Add user management
5. Add database persistence

---

**Status**: Ready for implementation

