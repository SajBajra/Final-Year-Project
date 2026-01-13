package com.lipika.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "payments")
public class Payment {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
    
    @Column(name = "transaction_uuid", unique = true, nullable = false)
    private String transactionUuid;
    
    @Column(name = "product_code")
    private String productCode;
    
    @Column(name = "product_name")
    private String productName;
    
    @Column(name = "amount", nullable = false)
    private Double amount;
    
    @Column(name = "tax_amount")
    private Double taxAmount;
    
    @Column(name = "service_charge")
    private Double serviceCharge;
    
    @Column(name = "delivery_charge")
    private Double deliveryCharge;
    
    @Column(name = "total_amount", nullable = false)
    private Double totalAmount;
    
    @Column(name = "status", nullable = false)
    @Enumerated(EnumType.STRING)
    private PaymentStatus status;
    
    @Column(name = "esewa_ref_id")
    private String esewaRefId;
    
    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;
    
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
    
    @Column(name = "verified_at")
    private LocalDateTime verifiedAt;
    
    // Enum for payment status
    public enum PaymentStatus {
        INITIATED,
        PENDING,
        COMPLETED,
        FAILED,
        REFUNDED
    }
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }
    
    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
    
    // Constructors
    public Payment() {
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public User getUser() {
        return user;
    }
    
    public void setUser(User user) {
        this.user = user;
    }
    
    public String getTransactionUuid() {
        return transactionUuid;
    }
    
    public void setTransactionUuid(String transactionUuid) {
        this.transactionUuid = transactionUuid;
    }
    
    public String getProductCode() {
        return productCode;
    }
    
    public void setProductCode(String productCode) {
        this.productCode = productCode;
    }
    
    public String getProductName() {
        return productName;
    }
    
    public void setProductName(String productName) {
        this.productName = productName;
    }
    
    public Double getAmount() {
        return amount;
    }
    
    public void setAmount(Double amount) {
        this.amount = amount;
    }
    
    public Double getTaxAmount() {
        return taxAmount;
    }
    
    public void setTaxAmount(Double taxAmount) {
        this.taxAmount = taxAmount;
    }
    
    public Double getServiceCharge() {
        return serviceCharge;
    }
    
    public void setServiceCharge(Double serviceCharge) {
        this.serviceCharge = serviceCharge;
    }
    
    public Double getDeliveryCharge() {
        return deliveryCharge;
    }
    
    public void setDeliveryCharge(Double deliveryCharge) {
        this.deliveryCharge = deliveryCharge;
    }
    
    public Double getTotalAmount() {
        return totalAmount;
    }
    
    public void setTotalAmount(Double totalAmount) {
        this.totalAmount = totalAmount;
    }
    
    public PaymentStatus getStatus() {
        return status;
    }
    
    public void setStatus(PaymentStatus status) {
        this.status = status;
    }
    
    public String getEsewaRefId() {
        return esewaRefId;
    }
    
    public void setEsewaRefId(String esewaRefId) {
        this.esewaRefId = esewaRefId;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
    
    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }
    
    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
    
    public LocalDateTime getVerifiedAt() {
        return verifiedAt;
    }
    
    public void setVerifiedAt(LocalDateTime verifiedAt) {
        this.verifiedAt = verifiedAt;
    }
}
