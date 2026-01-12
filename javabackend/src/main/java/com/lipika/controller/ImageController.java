package com.lipika.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@RestController
@RequestMapping("/api/images")
public class ImageController {
    
    private static final Logger log = LoggerFactory.getLogger(ImageController.class);
    
    /**
     * Serve image file
     * @param imagePath The relative path to the image (e.g., "uploads/ocr-images/20260112_123456_abc123.jpg")
     */
    @GetMapping("/**")
    public ResponseEntity<Resource> getImage(@RequestParam String path) {
        try {
            // Resolve the file path
            Path filePath = Paths.get(path).normalize();
            
            // Check if file exists
            if (!Files.exists(filePath)) {
                log.warn("Image not found: {}", path);
                return ResponseEntity.notFound().build();
            }
            
            // Load file as Resource
            Resource resource = new UrlResource(filePath.toUri());
            
            if (!resource.exists() || !resource.isReadable()) {
                log.warn("Image not readable: {}", path);
                return ResponseEntity.notFound().build();
            }
            
            // Determine content type
            String contentType = Files.probeContentType(filePath);
            if (contentType == null) {
                contentType = "application/octet-stream";
            }
            
            log.info("Serving image: {}", path);
            
            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(contentType))
                    .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + filePath.getFileName().toString() + "\"")
                    .body(resource);
                    
        } catch (IOException e) {
            log.error("Error serving image: {}", path, e);
            return ResponseEntity.internalServerError().build();
        }
    }
}

