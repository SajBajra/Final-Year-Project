package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.OCRRequest;
import com.lipika.model.OCRResponse;
import com.lipika.service.OCRService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequestMapping("/api/ocr")
@RequiredArgsConstructor
public class OCRController {
    
    private final OCRService ocrService;
    
    @PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ApiResponse<OCRResponse>> recognizeText(
            @RequestParam("image") MultipartFile image) {
        
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
        
        try {
            OCRResponse response = ocrService.recognizeText(image);
            
            if (response.isSuccess()) {
                return ResponseEntity.ok(ApiResponse.success("Text recognized successfully", response));
            } else {
                return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body(ApiResponse.error(response.getMessage() != null ? 
                                response.getMessage() : "OCR recognition failed"));
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
