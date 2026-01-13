package com.lipika.dto;

public class PaymentRequest {
    private Double amount;
    private String productName;
    private String productCode;
    
    public PaymentRequest() {
    }
    
    public PaymentRequest(Double amount, String productName, String productCode) {
        this.amount = amount;
        this.productName = productName;
        this.productCode = productCode;
    }
    
    // Getters and Setters
    public Double getAmount() {
        return amount;
    }
    
    public void setAmount(Double amount) {
        this.amount = amount;
    }
    
    public String getProductName() {
        return productName;
    }
    
    public void setProductName(String productName) {
        this.productName = productName;
    }
    
    public String getProductCode() {
        return productCode;
    }
    
    public void setProductCode(String productCode) {
        this.productCode = productCode;
    }
}
