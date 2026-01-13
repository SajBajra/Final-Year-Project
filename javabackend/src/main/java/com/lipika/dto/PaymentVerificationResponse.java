package com.lipika.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public class PaymentVerificationResponse {
    @JsonProperty("product_code")
    private String productCode;
    
    @JsonProperty("total_amount")
    private String totalAmount;
    
    @JsonProperty("transaction_uuid")
    private String transactionUuid;
    
    @JsonProperty("status")
    private String status;
    
    @JsonProperty("ref_id")
    private String refId;
    
    public PaymentVerificationResponse() {
    }
    
    // Getters and Setters
    public String getProductCode() {
        return productCode;
    }
    
    public void setProductCode(String productCode) {
        this.productCode = productCode;
    }
    
    public String getTotalAmount() {
        return totalAmount;
    }
    
    public void setTotalAmount(String totalAmount) {
        this.totalAmount = totalAmount;
    }
    
    public String getTransactionUuid() {
        return transactionUuid;
    }
    
    public void setTransactionUuid(String transactionUuid) {
        this.transactionUuid = transactionUuid;
    }
    
    public String getStatus() {
        return status;
    }
    
    public void setStatus(String status) {
        this.status = status;
    }
    
    public String getRefId() {
        return refId;
    }
    
    public void setRefId(String refId) {
        this.refId = refId;
    }
    
    public boolean isSuccessful() {
        return "COMPLETE".equalsIgnoreCase(status);
    }
}
