package com.lipika.controller;

import com.lipika.dto.PaymentInitiateResponse;
import com.lipika.dto.PaymentRequest;
import com.lipika.dto.PaymentVerificationResponse;
import com.lipika.model.ApiResponse;
import com.lipika.model.Payment;
import com.lipika.model.User;
import com.lipika.service.PaymentService;
import com.lipika.util.JwtUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/payment")
@CrossOrigin(origins = "*")
public class PaymentController {
    
    private static final Logger logger = LoggerFactory.getLogger(PaymentController.class);
    
    @Autowired
    private PaymentService paymentService;
    
    @Autowired
    private JwtUtil jwtUtil;
    
    /**
     * Initiate a payment transaction
     */
    @PostMapping("/initiate")
    public ResponseEntity<ApiResponse<PaymentInitiateResponse>> initiatePayment(
            @RequestBody PaymentRequest request,
            @RequestHeader("Authorization") String token) {
        try {
            // Extract user from token
            String jwt = token.substring(7);
            String username = jwtUtil.extractUsername(jwt);
            
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            User user = (User) authentication.getPrincipal();
            
            PaymentInitiateResponse response = paymentService.initiatePayment(request, user);
            
            return ResponseEntity.ok(new ApiResponse<>(
                true,
                "Payment initiated successfully",
                response
            ));
        } catch (Exception e) {
            logger.error("Error initiating payment: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiResponse<>(false, "Failed to initiate payment: " + e.getMessage(), null));
        }
    }
    
    /**
     * Verify a payment transaction
     */
    @GetMapping("/verify")
    public ResponseEntity<ApiResponse<PaymentVerificationResponse>> verifyPayment(
            @RequestParam("data") String data) {
        try {
            PaymentVerificationResponse response = paymentService.verifyPayment(data);
            
            return ResponseEntity.ok(new ApiResponse<>(
                true,
                "Payment verified successfully",
                response
            ));
        } catch (Exception e) {
            logger.error("Error verifying payment: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiResponse<>(false, "Failed to verify payment: " + e.getMessage(), null));
        }
    }
    
    /**
     * Get user's payment history
     */
    @GetMapping("/history")
    public ResponseEntity<ApiResponse<List<Payment>>> getPaymentHistory(
            @RequestHeader("Authorization") String token) {
        try {
            Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
            User user = (User) authentication.getPrincipal();
            
            List<Payment> payments = paymentService.getUserPayments(user);
            
            return ResponseEntity.ok(new ApiResponse<>(
                true,
                "Payment history retrieved successfully",
                payments
            ));
        } catch (Exception e) {
            logger.error("Error retrieving payment history: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiResponse<>(false, "Failed to retrieve payment history: " + e.getMessage(), null));
        }
    }
    
    /**
     * Get payment details by transaction UUID
     */
    @GetMapping("/{transactionUuid}")
    public ResponseEntity<ApiResponse<Payment>> getPaymentDetails(
            @PathVariable String transactionUuid) {
        try {
            Payment payment = paymentService.getPaymentByTransactionUuid(transactionUuid);
            
            return ResponseEntity.ok(new ApiResponse<>(
                true,
                "Payment details retrieved successfully",
                payment
            ));
        } catch (Exception e) {
            logger.error("Error retrieving payment details: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                .body(new ApiResponse<>(false, "Payment not found: " + e.getMessage(), null));
        }
    }
}
