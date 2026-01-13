package com.lipika.service.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.lipika.config.EsewaConfig;
import com.lipika.dto.PaymentInitiateResponse;
import com.lipika.dto.PaymentRequest;
import com.lipika.dto.PaymentVerificationResponse;
import com.lipika.model.Payment;
import com.lipika.model.User;
import com.lipika.repository.PaymentRepository;
import com.lipika.service.PaymentService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

@Service
public class PaymentServiceImpl implements PaymentService {
    
    private static final Logger logger = LoggerFactory.getLogger(PaymentServiceImpl.class);
    
    @Autowired
    private EsewaConfig esewaConfig;
    
    @Autowired
    private PaymentRepository paymentRepository;
    
    @Autowired
    private RestTemplate restTemplate;
    
    @Override
    @Transactional
    public PaymentInitiateResponse initiatePayment(PaymentRequest request, User user) {
        try {
            // Generate unique transaction UUID
            String transactionUuid = UUID.randomUUID().toString();
            
            // Calculate amounts
            Double amount = request.getAmount();
            Double taxAmount = Double.parseDouble(esewaConfig.getTaxAmount());
            Double serviceCharge = Double.parseDouble(esewaConfig.getServiceCharge());
            Double deliveryCharge = Double.parseDouble(esewaConfig.getDeliveryCharge());
            Double totalAmount = amount + taxAmount + serviceCharge + deliveryCharge;
            
            // Create payment record
            Payment payment = new Payment();
            payment.setUser(user);
            payment.setTransactionUuid(transactionUuid);
            payment.setProductCode(request.getProductCode());
            payment.setProductName(request.getProductName());
            payment.setAmount(amount);
            payment.setTaxAmount(taxAmount);
            payment.setServiceCharge(serviceCharge);
            payment.setDeliveryCharge(deliveryCharge);
            payment.setTotalAmount(totalAmount);
            payment.setStatus(Payment.PaymentStatus.INITIATED);
            
            paymentRepository.save(payment);
            
            // Generate signature
            String message = String.format(
                "total_amount=%s,transaction_uuid=%s,product_code=%s",
                totalAmount,
                transactionUuid,
                request.getProductCode()
            );
            
            String signature = generateSignature(message, esewaConfig.getSecretKey());
            
            // Create response
            PaymentInitiateResponse response = new PaymentInitiateResponse();
            response.setPaymentUrl(esewaConfig.getPaymentUrl());
            response.setTransactionUuid(transactionUuid);
            response.setAmount(String.valueOf(amount));
            response.setTaxAmount(String.valueOf(taxAmount));
            response.setTotalAmount(String.valueOf(totalAmount));
            response.setProductDeliveryCharge(String.valueOf(deliveryCharge));
            response.setProductServiceCharge(String.valueOf(serviceCharge));
            response.setProductCode(request.getProductCode());
            response.setSignature(signature);
            response.setSignedFieldNames("total_amount,transaction_uuid,product_code");
            
            logger.info("Payment initiated for user: {} with transaction UUID: {}", user.getEmail(), transactionUuid);
            
            return response;
        } catch (Exception e) {
            logger.error("Error initiating payment: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to initiate payment: " + e.getMessage());
        }
    }
    
    @Override
    @Transactional
    public PaymentVerificationResponse verifyPayment(String data) {
        try {
            // Decode the base64 data
            byte[] decodedBytes = Base64.getDecoder().decode(data);
            String decodedData = new String(decodedBytes, StandardCharsets.UTF_8);
            
            // Parse the decoded data (it's in JSON format)
            ObjectMapper mapper = new ObjectMapper();
            PaymentVerificationResponse verificationResponse = mapper.readValue(decodedData, PaymentVerificationResponse.class);
            
            // Call eSewa status check API
            String statusCheckUrl = esewaConfig.getStatusCheckUrl() + 
                "?product_code=" + verificationResponse.getProductCode() +
                "&total_amount=" + verificationResponse.getTotalAmount() +
                "&transaction_uuid=" + verificationResponse.getTransactionUuid();
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<String> entity = new HttpEntity<>(headers);
            
            ResponseEntity<PaymentVerificationResponse> response = restTemplate.exchange(
                statusCheckUrl,
                HttpMethod.GET,
                entity,
                PaymentVerificationResponse.class
            );
            
            PaymentVerificationResponse statusResponse = response.getBody();
            
            if (statusResponse != null && statusResponse.isSuccessful()) {
                // Update payment status
                updatePaymentStatus(
                    verificationResponse.getTransactionUuid(),
                    Payment.PaymentStatus.COMPLETED,
                    statusResponse.getRefId()
                );
                logger.info("Payment verified successfully: {}", verificationResponse.getTransactionUuid());
            } else {
                // Update payment status as failed
                updatePaymentStatus(
                    verificationResponse.getTransactionUuid(),
                    Payment.PaymentStatus.FAILED,
                    null
                );
                logger.warn("Payment verification failed: {}", verificationResponse.getTransactionUuid());
            }
            
            return statusResponse != null ? statusResponse : verificationResponse;
            
        } catch (Exception e) {
            logger.error("Error verifying payment: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to verify payment: " + e.getMessage());
        }
    }
    
    @Override
    public Payment getPaymentByTransactionUuid(String transactionUuid) {
        return paymentRepository.findByTransactionUuid(transactionUuid)
            .orElseThrow(() -> new RuntimeException("Payment not found with transaction UUID: " + transactionUuid));
    }
    
    @Override
    public List<Payment> getUserPayments(User user) {
        return paymentRepository.findByUserOrderByCreatedAtDesc(user);
    }
    
    @Override
    @Transactional
    public void updatePaymentStatus(String transactionUuid, Payment.PaymentStatus status, String refId) {
        Payment payment = getPaymentByTransactionUuid(transactionUuid);
        payment.setStatus(status);
        if (refId != null) {
            payment.setEsewaRefId(refId);
        }
        if (status == Payment.PaymentStatus.COMPLETED) {
            payment.setVerifiedAt(LocalDateTime.now());
        }
        paymentRepository.save(payment);
    }
    
    /**
     * Generate HMAC-SHA256 signature for eSewa payment
     */
    private String generateSignature(String message, String secret) {
        try {
            Mac sha256Hmac = Mac.getInstance("HmacSHA256");
            SecretKeySpec secretKey = new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256");
            sha256Hmac.init(secretKey);
            byte[] signedBytes = sha256Hmac.doFinal(message.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(signedBytes);
        } catch (Exception e) {
            logger.error("Error generating signature: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to generate payment signature");
        }
    }
}
