package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.OCRRequest;
import com.lipika.model.OCRResponse;
import com.lipika.service.OCRService;
import com.lipika.util.JwtUtil;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/ocr")
public class OCRController {
    
    private static final Logger log = LoggerFactory.getLogger(OCRController.class);
    
    private final OCRService ocrService;
    private final JwtUtil jwtUtil;
    
    public OCRController(OCRService ocrService, JwtUtil jwtUtil) {
        this.ocrService = ocrService;
        this.jwtUtil = jwtUtil;
    }
    
    @PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ApiResponse<OCRResponse>> recognizeText(
            @RequestParam("image") MultipartFile image,
            HttpServletRequest request) {
        
        log.info("Received OCR request for image: {}", image.getOriginalFilename());
        
        // Validate file
        if (image.isEmpty()) {
            return ResponseEntity.badRequest()
                    .body(ApiResponse.error("Image file is required"));
        }
        
        // Validate file type
        String contentType = image.getContentType();
        if (contentType == null || 
            (!contentType.startsWith("image/") && 
             !contentType.equals("application/octet-stream"))) {
            return ResponseEntity.badRequest()
                    .body(ApiResponse.error("File must be an image"));
        }
        
        // Extract user ID from authentication if present
        Long userId = null;
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        if (authentication != null && authentication.isAuthenticated() && 
            !authentication.getName().equals("anonymousUser")) {
            try {
                // Try to extract from JWT token in Authorization header
                String authHeader = request.getHeader("Authorization");
                if (authHeader != null && authHeader.startsWith("Bearer ")) {
                    String token = authHeader.substring(7);
                    userId = jwtUtil.extractUserId(token);
                }
            } catch (Exception e) {
                log.debug("Could not extract user ID from token", e);
            }
        }
        
        try {
            OCRResponse response = ocrService.recognizeText(image, request, userId);
            
            if (response.isSuccess()) {
                return ResponseEntity.ok(ApiResponse.success("Text recognized successfully", response));
            } else {
                // Check if it's a trial limit error
                if (response.getTrialInfo() != null && response.getTrialInfo().getRequiresLogin()) {
                    return ResponseEntity.status(HttpStatus.FORBIDDEN)
                            .body(ApiResponse.error(response.getMessage() != null ? 
                                    response.getMessage() : "Trial limit exceeded", response));
                }
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body(ApiResponse.error(response.getMessage() != null ? 
                                response.getMessage() : "OCR recognition failed", response));
            }
            
        } catch (Exception e) {
            log.error("Error processing OCR request", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(ApiResponse.error("Error processing OCR request: " + e.getMessage()));
        }
    }
    
    @GetMapping("/health")
    public ResponseEntity<ApiResponse<String>> health() {
        return ResponseEntity.ok(ApiResponse.success("OCR service is healthy"));
    }
}
