package com.lipika.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

@Configuration
@PropertySource("classpath:application-payment.properties")
@ConfigurationProperties(prefix = "esewa")
public class EsewaConfig {
    
    private Merchant merchant = new Merchant();
    private String secretKey;
    private String paymentUrl;
    private String statusCheckUrl;
    private String successUrl;
    private String failureUrl;
    private String taxAmount;
    private String serviceCharge;
    private String deliveryCharge;
    
    // Nested class for merchant configuration
    public static class Merchant {
        private String code;
        
        public String getCode() {
            return code;
        }
        
        public void setCode(String code) {
            this.code = code;
        }
    }
    
    // Getters and Setters
    public Merchant getMerchant() {
        return merchant;
    }
    
    public void setMerchant(Merchant merchant) {
        this.merchant = merchant;
    }
    
    public String getSecretKey() {
        return secretKey;
    }
    
    public void setSecretKey(String secretKey) {
        this.secretKey = secretKey;
    }
    
    public String getPaymentUrl() {
        return paymentUrl;
    }
    
    public void setPaymentUrl(String paymentUrl) {
        this.paymentUrl = paymentUrl;
    }
    
    public String getStatusCheckUrl() {
        return statusCheckUrl;
    }
    
    public void setStatusCheckUrl(String statusCheckUrl) {
        this.statusCheckUrl = statusCheckUrl;
    }
    
    public String getSuccessUrl() {
        return successUrl;
    }
    
    public void setSuccessUrl(String successUrl) {
        this.successUrl = successUrl;
    }
    
    public String getFailureUrl() {
        return failureUrl;
    }
    
    public void setFailureUrl(String failureUrl) {
        this.failureUrl = failureUrl;
    }
    
    public String getTaxAmount() {
        return taxAmount;
    }
    
    public void setTaxAmount(String taxAmount) {
        this.taxAmount = taxAmount;
    }
    
    public String getServiceCharge() {
        return serviceCharge;
    }
    
    public void setServiceCharge(String serviceCharge) {
        this.serviceCharge = serviceCharge;
    }
    
    public String getDeliveryCharge() {
        return deliveryCharge;
    }
    
    public void setDeliveryCharge(String deliveryCharge) {
        this.deliveryCharge = deliveryCharge;
    }
}
