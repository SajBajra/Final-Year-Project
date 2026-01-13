package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.service.EmailService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/contact")
public class ContactController {
    
    private static final Logger logger = LoggerFactory.getLogger(ContactController.class);
    
    @Autowired
    private EmailService emailService;
    
    @PostMapping("/submit")
    public ResponseEntity<ApiResponse<String>> submitContactForm(@RequestBody Map<String, String> request) {
        try {
            String name = request.get("name");
            String email = request.get("email");
            String subject = request.get("subject");
            String message = request.get("message");
            
            // Validate inputs
            if (name == null || name.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(new ApiResponse<>(false, "Name is required", null));
            }
            
            if (email == null || email.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(new ApiResponse<>(false, "Email is required", null));
            }
            
            if (subject == null || subject.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(new ApiResponse<>(false, "Subject is required", null));
            }
            
            if (message == null || message.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(new ApiResponse<>(false, "Message is required", null));
            }
            
            // Send email
            emailService.sendContactFormEmail(name, email, subject, message);
            
            logger.info("Contact form submitted by: {} ({})", name, email);
            
            return ResponseEntity.ok(new ApiResponse<>(true, 
                "Thank you for contacting us! We'll get back to you soon.", null));
                
        } catch (Exception e) {
            logger.error("Error processing contact form: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiResponse<>(false, "Failed to send message: " + e.getMessage(), null));
        }
    }
}
