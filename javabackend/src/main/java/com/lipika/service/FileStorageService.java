package com.lipika.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

@Service
public class FileStorageService {
    
    private static final Logger log = LoggerFactory.getLogger(FileStorageService.class);
    
    @Value("${file.upload-dir:uploads/ocr-images}")
    private String uploadDir;
    
    /**
     * Save uploaded image file to local storage
     * @param file The uploaded image file
     * @return The relative path to the saved file
     */
    public String saveImage(MultipartFile file) throws IOException {
        // Create upload directory if it doesn't exist
        Path uploadPath = Paths.get(uploadDir);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
            log.info("Created upload directory: {}", uploadPath.toAbsolutePath());
        }
        
        // Generate unique filename with timestamp
        String originalFilename = file.getOriginalFilename();
        String extension = "";
        if (originalFilename != null && originalFilename.contains(".")) {
            extension = originalFilename.substring(originalFilename.lastIndexOf("."));
        }
        
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String uniqueId = UUID.randomUUID().toString().substring(0, 8);
        String newFilename = String.format("%s_%s%s", timestamp, uniqueId, extension);
        
        // Save file
        Path targetPath = uploadPath.resolve(newFilename);
        Files.copy(file.getInputStream(), targetPath, StandardCopyOption.REPLACE_EXISTING);
        
        log.info("Saved image file: {}", newFilename);
        
        // Return relative path
        return uploadDir + "/" + newFilename;
    }
    
    /**
     * Save byte array as image file
     * @param bytes The image byte array
     * @param filename The original filename
     * @return The relative path to the saved file
     */
    public String saveImageBytes(byte[] bytes, String filename) throws IOException {
        // Create upload directory if it doesn't exist
        Path uploadPath = Paths.get(uploadDir);
        if (!Files.exists(uploadPath)) {
            Files.createDirectories(uploadPath);
            log.info("Created upload directory: {}", uploadPath.toAbsolutePath());
        }
        
        // Generate unique filename with timestamp
        String extension = "";
        if (filename != null && filename.contains(".")) {
            extension = filename.substring(filename.lastIndexOf("."));
        }
        
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String uniqueId = UUID.randomUUID().toString().substring(0, 8);
        String newFilename = String.format("%s_%s%s", timestamp, uniqueId, extension);
        
        // Save file
        Path targetPath = uploadPath.resolve(newFilename);
        Files.write(targetPath, bytes);
        
        log.info("Saved image file from bytes: {}", newFilename);
        
        // Return relative path
        return uploadDir + "/" + newFilename;
    }
    
    /**
     * Get the absolute path of the upload directory
     */
    public Path getUploadPath() {
        return Paths.get(uploadDir).toAbsolutePath();
    }
}

