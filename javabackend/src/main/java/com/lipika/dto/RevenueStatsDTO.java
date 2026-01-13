package com.lipika.dto;

public class RevenueStatsDTO {
    private Double totalRevenue;
    private Long totalTransactions;
    private Double monthlyRevenue;
    private Long monthlyTransactions;
    private Long pendingTransactions;
    private Long completedTransactions;
    private Long failedTransactions;
    
    public RevenueStatsDTO() {
    }
    
    public RevenueStatsDTO(Double totalRevenue, Long totalTransactions, Double monthlyRevenue, 
                          Long monthlyTransactions, Long pendingTransactions, 
                          Long completedTransactions, Long failedTransactions) {
        this.totalRevenue = totalRevenue;
        this.totalTransactions = totalTransactions;
        this.monthlyRevenue = monthlyRevenue;
        this.monthlyTransactions = monthlyTransactions;
        this.pendingTransactions = pendingTransactions;
        this.completedTransactions = completedTransactions;
        this.failedTransactions = failedTransactions;
    }
    
    // Getters and Setters
    public Double getTotalRevenue() {
        return totalRevenue;
    }
    
    public void setTotalRevenue(Double totalRevenue) {
        this.totalRevenue = totalRevenue;
    }
    
    public Long getTotalTransactions() {
        return totalTransactions;
    }
    
    public void setTotalTransactions(Long totalTransactions) {
        this.totalTransactions = totalTransactions;
    }
    
    public Double getMonthlyRevenue() {
        return monthlyRevenue;
    }
    
    public void setMonthlyRevenue(Double monthlyRevenue) {
        this.monthlyRevenue = monthlyRevenue;
    }
    
    public Long getMonthlyTransactions() {
        return monthlyTransactions;
    }
    
    public void setMonthlyTransactions(Long monthlyTransactions) {
        this.monthlyTransactions = monthlyTransactions;
    }
    
    public Long getPendingTransactions() {
        return pendingTransactions;
    }
    
    public void setPendingTransactions(Long pendingTransactions) {
        this.pendingTransactions = pendingTransactions;
    }
    
    public Long getCompletedTransactions() {
        return completedTransactions;
    }
    
    public void setCompletedTransactions(Long completedTransactions) {
        this.completedTransactions = completedTransactions;
    }
    
    public Long getFailedTransactions() {
        return failedTransactions;
    }
    
    public void setFailedTransactions(Long failedTransactions) {
        this.failedTransactions = failedTransactions;
    }
}
