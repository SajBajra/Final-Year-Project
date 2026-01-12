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
    
    @Column(name = "image_path", length = 1000)
    private String imagePath;
    
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
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public Long getUserId() {
        return userId;
    }
    
    public void setUserId(Long userId) {
        this.userId = userId;
    }
    
    public Boolean getIsRegistered() {
        return isRegistered;
    }
    
    public void setIsRegistered(Boolean isRegistered) {
        this.isRegistered = isRegistered;
    }
    
    public String getIpAddress() {
        return ipAddress;
    }
    
    public void setIpAddress(String ipAddress) {
        this.ipAddress = ipAddress;
    }
    
    public String getCookieId() {
        return cookieId;
    }
    
    public void setCookieId(String cookieId) {
        this.cookieId = cookieId;
    }
    
    public String getImageFilename() {
        return imageFilename;
    }
    
    public void setImageFilename(String imageFilename) {
        this.imageFilename = imageFilename;
    }
    
    public String getImagePath() {
        return imagePath;
    }
    
    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }
    
    public String getRecognizedText() {
        return recognizedText;
    }
    
    public void setRecognizedText(String recognizedText) {
        this.recognizedText = recognizedText;
    }
    
    public Integer getCharacterCount() {
        return characterCount;
    }
    
    public void setCharacterCount(Integer characterCount) {
        this.characterCount = characterCount;
    }
    
    public Double getConfidence() {
        return confidence;
    }
    
    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    
    public String getLanguage() {
        return language;
    }
    
    public void setLanguage(String language) {
        this.language = language;
    }
    
    public User getUser() {
        return user;
    }
    
    public void setUser(User user) {
        this.user = user;
    }
}
