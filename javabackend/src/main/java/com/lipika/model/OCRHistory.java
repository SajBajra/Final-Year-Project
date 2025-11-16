package com.lipika.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
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
    
    @Column(name = "user_id")
    private Long userId;
    
    @Column(name = "is_registered", nullable = false)
    private Boolean isRegistered = false;
    
    @Column(name = "ip_address", length = 45)
    private String ipAddress;
    
    @Column(name = "cookie_id", length = 255)
    private String cookieId;
    
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
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", insertable = false, updatable = false)
    @JsonIgnore // Prevent lazy loading issues during JSON serialization
    private User user;
    
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
