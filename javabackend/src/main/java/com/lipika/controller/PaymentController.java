package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.Payment;
import com.lipika.model.User;
import com.lipika.repository.PaymentRepository;
import com.lipika.repository.UserRepository;
import com.lipika.util.JwtUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.Base64;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/payment")
public class PaymentController {

    private static final Logger logger = LoggerFactory.getLogger(PaymentController.class);

    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private PaymentRepository paymentRepository;

    @Autowired
    private JwtUtil jwtUtil;

    @Value("${esewa.secret-key}")
    private String esewaSecretKey;
    
    @Value("${esewa.test-mode:true}")
    private boolean esewaTestMode;

    /**
     * Verify payment from eSewa and upgrade user.
     * This endpoint is called after eSewa redirects to success URL.
     */
    @PostMapping("/verify")
    public ResponseEntity<ApiResponse<String>> verifyPayment(
            @RequestBody Map<String, Object> requestData,
            @RequestHeader("Authorization") String token) {
        try {
            logger.info("=== PAYMENT VERIFICATION REQUEST ===");
            logger.info("Request data: {}", requestData);

            // Extract username from token
            String jwt = token.substring(7);
            String username = jwtUtil.extractUsername(jwt);

            // Get user from database
            Optional<User> userOptional = userRepository.findByUsername(username);
            if (userOptional.isEmpty()) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND)
                        .body(new ApiResponse<>(false, "User not found: " + username, null));
            }
            User user = userOptional.get();
            logger.info("User found: {}", user.getUsername());

            // Extract payment data from eSewa response
            @SuppressWarnings("unchecked")
            Map<String, Object> paymentData = (Map<String, Object>) requestData.get("paymentData");
            
            if (paymentData == null) {
                return ResponseEntity.badRequest()
                        .body(new ApiResponse<>(false, "Payment data missing", null));
            }

            // Log all payment data fields for debugging
            logger.info("=== Payment Data Fields ===");
            paymentData.forEach((key, value) -> logger.info("{} = {}", key, value));
            logger.info("===========================");
            
            // Extract fields from eSewa response
            String status = (String) paymentData.get("status");
            String transactionCode = (String) paymentData.get("transaction_code");
            String totalAmount = String.valueOf(paymentData.get("total_amount"));
            String transactionUuid = (String) paymentData.get("transaction_uuid");
            String productCode = (String) paymentData.get("product_code");
            String signedFieldNames = (String) paymentData.get("signed_field_names");
            String receivedSignature = (String) paymentData.get("signature");

            logger.info("Extracted values:");
            logger.info("Payment status: {}", status);
            logger.info("Transaction code: {}", transactionCode);
            logger.info("Total amount: {}", totalAmount);
            logger.info("Transaction UUID: {}", transactionUuid);
            logger.info("Product code: {}", productCode);
            logger.info("Signed field names: {}", signedFieldNames);
            logger.info("Received signature: {}", receivedSignature);

            // Verify payment status
            if (!"COMPLETE".equalsIgnoreCase(status)) {
                return ResponseEntity.badRequest()
                        .body(new ApiResponse<>(false, "Payment not completed", null));
            }

            // Generate expected signature for verification
            String message = "total_amount=" + totalAmount + 
                           ",transaction_uuid=" + transactionUuid + 
                           ",product_code=" + productCode;
            String expectedSignature = generateEsewaSignature(message);

            // Debug logging
            logger.info("Verification message: {}", message);
            logger.info("Received signature: {}", receivedSignature);
            logger.info("Expected signature: {}", expectedSignature);

            // Verify signature matches
            boolean signatureValid = expectedSignature.equals(receivedSignature);
            logger.info("Signature validation result: {}", signatureValid);
            logger.info("eSewa test mode enabled: {}", esewaTestMode);
            
            if (!signatureValid) {
                logger.warn("Signature mismatch for user: {}", username);
                
                if (esewaTestMode) {
                    // In test mode, allow if status is COMPLETE
                    logger.warn("Running in TEST MODE - will allow payment if status is COMPLETE");
                    if (!"COMPLETE".equalsIgnoreCase(status)) {
                        return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                                .body(new ApiResponse<>(false, "Invalid payment signature and status not complete", null));
                    }
                    logger.warn("Allowing payment despite signature mismatch (TEST MODE - status is COMPLETE)");
                } else {
                    // In production mode, strict signature verification
                    logger.error("PRODUCTION MODE: Rejecting payment due to invalid signature");
                    return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                            .body(new ApiResponse<>(false, "Invalid payment signature", null));
                }
            } else {
                logger.info("✓ Signature verification passed");
            }

            // Check if payment already processed
            Optional<Payment> existingPayment = paymentRepository.findByTransactionUuid(transactionUuid);
            if (existingPayment.isPresent() && "COMPLETED".equals(existingPayment.get().getStatus())) {
                logger.info("Payment already processed for transaction: {}", transactionUuid);
                return ResponseEntity.ok(new ApiResponse<>(true, "Payment already processed", null));
            }
            
            // Determine plan type from total amount
            String planType = "monthly";
            Double amount = Double.parseDouble(totalAmount);
            if (amount >= 1000) {
                planType = "yearly";
            }
            
            // Create payment record
            Payment payment = new Payment();
            payment.setTransactionUuid(transactionUuid);
            payment.setTransactionCode(transactionCode);
            payment.setUserId(user.getId());
            payment.setUsername(username);
            payment.setPlanType(planType);
            payment.setAmount(amount);
            payment.setStatus("COMPLETED");
            payment.setPaymentMethod("eSewa");
            payment.setVerifiedAt(LocalDateTime.now());
            paymentRepository.save(payment);
            
            // Payment verified, upgrade user to premium
            user.setRole("PREMIUM");
            user.setIsPremium(true);
            userRepository.save(user);

            logger.info("User {} upgraded to PREMIUM successfully via eSewa transaction: {} with amount: NPR {}", 
                       username, transactionCode, amount);
            return ResponseEntity.ok(new ApiResponse<>(true, "Payment verified and user upgraded to Premium", null));

        } catch (Exception e) {
            logger.error("Error verifying payment: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(new ApiResponse<>(false, "Failed to verify payment: " + e.getMessage(), null));
        }
    }

    /**
     * Generates eSewa HMAC-SHA256 signature.
     */
    private String generateEsewaSignature(String message) {
        try {
            Mac sha256Hmac = Mac.getInstance("HmacSHA256");
            SecretKeySpec secretKeySpec = new SecretKeySpec(esewaSecretKey.getBytes(StandardCharsets.UTF_8), "HmacSHA256");
            sha256Hmac.init(secretKeySpec);
            byte[] hash = sha256Hmac.doFinal(message.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(hash);
        } catch (Exception e) {
            logger.error("Error generating eSewa signature: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to generate signature", e);
        }
    }
}
