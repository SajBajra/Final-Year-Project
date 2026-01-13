package com.lipika.dto;

import java.time.LocalDateTime;

public class AuthResponse {
    private String token;
    private String type = "Bearer";
    private Long userId;
    private String username;
    private String email;
    private String role;
    private Integer remainingTrials;
    private Boolean isPremium;
    private LocalDateTime premiumUntil;
    
    // Constructors
    public AuthResponse() {
    }
    
    public AuthResponse(String token, String type, Long userId, String username, String email, String role, Integer remainingTrials) {
        this.token = token;
        this.type = type;
        this.userId = userId;
        this.username = username;
        this.email = email;
        this.role = role;
        this.remainingTrials = remainingTrials;
    }
    
    public AuthResponse(String token, String type, Long userId, String username, String email, String role, Integer remainingTrials, Boolean isPremium, LocalDateTime premiumUntil) {
        this.token = token;
        this.type = type;
        this.userId = userId;
        this.username = username;
        this.email = email;
        this.role = role;
        this.remainingTrials = remainingTrials;
        this.isPremium = isPremium;
        this.premiumUntil = premiumUntil;
    }
    
    // Getters and Setters
    public String getToken() {
        return token;
    }
    
    public void setToken(String token) {
        this.token = token;
    }
    
    public String getType() {
        return type;
    }
    
    public void setType(String type) {
        this.type = type;
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
    
    public String getEmail() {
        return email;
    }
    
    public void setEmail(String email) {
        this.email = email;
    }
    
    public String getRole() {
        return role;
    }
    
    public void setRole(String role) {
        this.role = role;
    }
    
    public Integer getRemainingTrials() {
        return remainingTrials;
    }
    
    public void setRemainingTrials(Integer remainingTrials) {
        this.remainingTrials = remainingTrials;
    }
    
    public Boolean getIsPremium() {
        return isPremium;
    }
    
    public void setIsPremium(Boolean isPremium) {
        this.isPremium = isPremium;
    }
    
    public LocalDateTime getPremiumUntil() {
        return premiumUntil;
    }
    
    public void setPremiumUntil(LocalDateTime premiumUntil) {
        this.premiumUntil = premiumUntil;
    }
}

