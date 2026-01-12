package com.lipika.dto;

import java.time.LocalDateTime;

public class OCRHistoryDTO {
    private Long id;
    private Long userId;
    private String username;
    private String userEmail;
    private String userRole;
    private Boolean isRegistered;
    private String ipAddress;
    private String cookieId;
    private String imageFilename;
    private String imagePath;
    private String recognizedText;
    private Integer characterCount;
    private Double confidence;
    private LocalDateTime timestamp;
    private String language;
    
    // Constructors
    public OCRHistoryDTO() {}
    
    public OCRHistoryDTO(Long id, Long userId, String username, String userEmail, String userRole,
                         Boolean isRegistered, String ipAddress, String cookieId,
                         String imageFilename, String imagePath, String recognizedText,
                         Integer characterCount, Double confidence, LocalDateTime timestamp, String language) {
        this.id = id;
        this.userId = userId;
        this.username = username;
        this.userEmail = userEmail;
        this.userRole = userRole;
        this.isRegistered = isRegistered;
        this.ipAddress = ipAddress;
        this.cookieId = cookieId;
        this.imageFilename = imageFilename;
        this.imagePath = imagePath;
        this.recognizedText = recognizedText;
        this.characterCount = characterCount;
        this.confidence = confidence;
        this.timestamp = timestamp;
        this.language = language;
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
    
    public String getUsername() {
        return username;
    }
    
    public void setUsername(String username) {
        this.username = username;
    }
    
    public String getUserEmail() {
        return userEmail;
    }
    
    public void setUserEmail(String userEmail) {
        this.userEmail = userEmail;
    }
    
    public String getUserRole() {
        return userRole;
    }
    
    public void setUserRole(String userRole) {
        this.userRole = userRole;
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
}

