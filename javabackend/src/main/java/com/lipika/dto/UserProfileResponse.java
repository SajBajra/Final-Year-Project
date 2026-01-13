package com.lipika.dto;

import java.time.LocalDateTime;

public class UserProfileResponse {
    private Long id;
    private String username;
    private String email;
    private String role;
    private Integer usageCount;
    private Integer usageLimit;
    private Boolean isPremium;
    private String planType; // "monthly" or "yearly" for premium users
    private LocalDateTime premiumUntil;
    private LocalDateTime createdAt;
    private LocalDateTime lastLogin;
    
    public UserProfileResponse() {
    }
    
    public UserProfileResponse(Long id, String username, String email, String role,
                              Integer usageCount, Integer usageLimit, Boolean isPremium,
                              LocalDateTime premiumUntil, LocalDateTime createdAt, LocalDateTime lastLogin) {
        this.id = id;
        this.username = username;
        this.email = email;
        this.role = role;
        this.usageCount = usageCount;
        this.usageLimit = usageLimit;
        this.isPremium = isPremium;
        this.premiumUntil = premiumUntil;
        this.createdAt = createdAt;
        this.lastLogin = lastLogin;
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
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
    
    public Integer getUsageCount() {
        return usageCount;
    }
    
    public void setUsageCount(Integer usageCount) {
        this.usageCount = usageCount;
    }
    
    public Integer getUsageLimit() {
        return usageLimit;
    }
    
    public void setUsageLimit(Integer usageLimit) {
        this.usageLimit = usageLimit;
    }
    
    public Boolean getIsPremium() {
        return isPremium;
    }
    
    public void setIsPremium(Boolean isPremium) {
        this.isPremium = isPremium;
    }
    
    public String getPlanType() {
        return planType;
    }
    
    public void setPlanType(String planType) {
        this.planType = planType;
    }
    
    public LocalDateTime getPremiumUntil() {
        return premiumUntil;
    }
    
    public void setPremiumUntil(LocalDateTime premiumUntil) {
        this.premiumUntil = premiumUntil;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
    
    public LocalDateTime getLastLogin() {
        return lastLogin;
    }
    
    public void setLastLogin(LocalDateTime lastLogin) {
        this.lastLogin = lastLogin;
    }
}

