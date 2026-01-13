package com.lipika.dto;

public class PaymentInitiateResponse {
    private String paymentUrl;
    private String transactionUuid;
    private String amount;
    private String taxAmount;
    private String totalAmount;
    private String productDeliveryCharge;
    private String productServiceCharge;
    private String productCode;
    private String signature;
    private String signedFieldNames;
    
    public PaymentInitiateResponse() {
    }
    
    // Getters and Setters
    public String getPaymentUrl() {
        return paymentUrl;
    }
    
    public void setPaymentUrl(String paymentUrl) {
        this.paymentUrl = paymentUrl;
    }
    
    public String getTransactionUuid() {
        return transactionUuid;
    }
    
    public void setTransactionUuid(String transactionUuid) {
        this.transactionUuid = transactionUuid;
    }
    
    public String getAmount() {
        return amount;
    }
    
    public void setAmount(String amount) {
        this.amount = amount;
    }
    
    public String getTaxAmount() {
        return taxAmount;
    }
    
    public void setTaxAmount(String taxAmount) {
        this.taxAmount = taxAmount;
    }
    
    public String getTotalAmount() {
        return totalAmount;
    }
    
    public void setTotalAmount(String totalAmount) {
        this.totalAmount = totalAmount;
    }
    
    public String getProductDeliveryCharge() {
        return productDeliveryCharge;
    }
    
    public void setProductDeliveryCharge(String productDeliveryCharge) {
        this.productDeliveryCharge = productDeliveryCharge;
    }
    
    public String getProductServiceCharge() {
        return productServiceCharge;
    }
    
    public void setProductServiceCharge(String productServiceCharge) {
        this.productServiceCharge = productServiceCharge;
    }
    
    public String getProductCode() {
        return productCode;
    }
    
    public void setProductCode(String productCode) {
        this.productCode = productCode;
    }
    
    public String getSignature() {
        return signature;
    }
    
    public void setSignature(String signature) {
        this.signature = signature;
    }
    
    public String getSignedFieldNames() {
        return signedFieldNames;
    }
    
    public void setSignedFieldNames(String signedFieldNames) {
        this.signedFieldNames = signedFieldNames;
    }
}
