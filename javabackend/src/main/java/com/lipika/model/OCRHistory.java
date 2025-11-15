package com.lipika.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "ocr_history")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class OCRHistory {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "image_filename", length = 500)
    private String imageFilename;
    
    @Column(name = "recognized_text", columnDefinition = "TEXT")
    private String recognizedText;
    
    @Column(name = "character_count")
    private Integer characterCount;
    
    @Column(name = "confidence")
    private Double confidence;
    
    @Column(name = "timestamp", nullable = false)
    private LocalDateTime timestamp;
    
    @Column(name = "language", length = 50)
    private String language;
    
    @PrePersist
    protected void onCreate() {
        if (timestamp == null) {
            timestamp = LocalDateTime.now();
        }
        if (language == null) {
            language = "devanagari";
        }
    }
}
