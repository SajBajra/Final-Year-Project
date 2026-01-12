package com.lipika.dto;

public class TrialInfo {
    private Integer remainingTrials;
    private Integer usedTrials;
    private Integer maxTrials;
    private Boolean requiresLogin;
    
    // Constructors
    public TrialInfo() {
    }
    
    public TrialInfo(Integer remainingTrials, Integer usedTrials, Integer maxTrials, Boolean requiresLogin) {
        this.remainingTrials = remainingTrials;
        this.usedTrials = usedTrials;
        this.maxTrials = maxTrials;
        this.requiresLogin = requiresLogin;
    }
    
    // Getters and Setters
    public Integer getRemainingTrials() {
        return remainingTrials;
    }
    
    public void setRemainingTrials(Integer remainingTrials) {
        this.remainingTrials = remainingTrials;
    }
    
    public Integer getUsedTrials() {
        return usedTrials;
    }
    
    public void setUsedTrials(Integer usedTrials) {
        this.usedTrials = usedTrials;
    }
    
    public Integer getMaxTrials() {
        return maxTrials;
    }
    
    public void setMaxTrials(Integer maxTrials) {
        this.maxTrials = maxTrials;
    }
    
    public Boolean getRequiresLogin() {
        return requiresLogin;
    }
    
    public void setRequiresLogin(Boolean requiresLogin) {
        this.requiresLogin = requiresLogin;
    }
}

