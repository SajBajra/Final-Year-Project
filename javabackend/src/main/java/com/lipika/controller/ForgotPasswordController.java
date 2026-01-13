package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.OTP;
import com.lipika.model.User;
import com.lipika.repository.OTPRepository;
import com.lipika.repository.UserRepository;
import com.lipika.service.EmailService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

@RestController
@RequestMapping("/api/auth")
public class ForgotPasswordController {
    
    private static final Logger logger = LoggerFactory.getLogger(ForgotPasswordController.class);
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private OTPRepository otpRepository;
    
    @Autowired
    private EmailService emailService;
    
    @Autowired
    private PasswordEncoder passwordEncoder;
    
    @Value("${otp.expiry.minutes:5}")
    private int otpExpiryMinutes;
    
    /**
     * Request OTP for password reset
     * POST /api/auth/forgot-password
     */
    @PostMapping("/forgot-password")
    @Transactional
    public ResponseEntity<ApiResponse<String>> requestPasswordReset(@RequestBody Map<String, String> request) {
        try {
            String email = request.get("email");
            
            if (email == null || email.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(new ApiResponse<>(false, "Email is required", null));
            }
            
            // Check if user exists
            Optional<User> userOptional = userRepository.findByEmail(email);
            if (userOptional.isEmpty()) {
                // Don't reveal if email exists or not for security
                return ResponseEntity.ok(new ApiResponse<>(true, 
                    "If an account exists with this email, an OTP has been sent", null));
            }
            
            // Generate 6-digit OTP
            String otpCode = String.format("%06d", new Random().nextInt(999999));
            
            // Delete any existing OTPs for this email
            otpRepository.deleteByEmail(email);
            
            // Create new OTP
            OTP otp = new OTP();
            otp.setEmail(email);
            otp.setOtpCode(otpCode);
            otp.setCreatedAt(LocalDateTime.now());
            otp.setExpiresAt(LocalDateTime.now().plusMinutes(otpExpiryMinutes));
            otp.setVerified(false);
            
            otpRepository.save(otp);
            
            // Send OTP email
            emailService.sendOTPEmail(email, otpCode);
            
            logger.info("OTP generated and sent to email: {}", email);
            
            return ResponseEntity.ok(new ApiResponse<>(true, 
                "OTP has been sent to your email address", null));
                
        } catch (Exception e) {
            logger.error("Error in forgot password request: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiResponse<>(false, "Failed to process request: " + e.getMessage(), null));
        }
    }
    
    /**
     * Verify OTP
     * POST /api/auth/verify-otp
     */
    @PostMapping("/verify-otp")
    public ResponseEntity<ApiResponse<String>> verifyOTP(@RequestBody Map<String, String> request) {
        try {
            String email = request.get("email");
            String otpCode = request.get("otp");
            
            if (email == null || email.trim().isEmpty() || otpCode == null || otpCode.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(new ApiResponse<>(false, "Email and OTP are required", null));
            }
            
            // Find OTP
            Optional<OTP> otpOptional = otpRepository.findByEmailAndOtpCodeAndVerifiedFalse(email, otpCode);
            
            if (otpOptional.isEmpty()) {
                return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(new ApiResponse<>(false, "Invalid OTP", null));
            }
            
            OTP otp = otpOptional.get();
            
            // Check if OTP is expired
            if (otp.isExpired()) {
                return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(new ApiResponse<>(false, "OTP has expired. Please request a new one", null));
            }
            
            // Mark OTP as verified
            otp.setVerified(true);
            otpRepository.save(otp);
            
            logger.info("OTP verified successfully for email: {}", email);
            
            return ResponseEntity.ok(new ApiResponse<>(true, "OTP verified successfully", null));
            
        } catch (Exception e) {
            logger.error("Error verifying OTP: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiResponse<>(false, "Failed to verify OTP: " + e.getMessage(), null));
        }
    }
    
    /**
     * Reset password after OTP verification
     * POST /api/auth/reset-password
     */
    @PostMapping("/reset-password")
    @Transactional
    public ResponseEntity<ApiResponse<String>> resetPassword(@RequestBody Map<String, String> request) {
        try {
            String email = request.get("email");
            String otpCode = request.get("otp");
            String newPassword = request.get("newPassword");
            
            if (email == null || email.trim().isEmpty() || 
                otpCode == null || otpCode.trim().isEmpty() ||
                newPassword == null || newPassword.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                    .body(new ApiResponse<>(false, "Email, OTP, and new password are required", null));
            }
            
            // Validate password length
            if (newPassword.length() < 6) {
                return ResponseEntity.badRequest()
                    .body(new ApiResponse<>(false, "Password must be at least 6 characters long", null));
            }
            
            // Find verified OTP
            Optional<OTP> otpOptional = otpRepository.findByEmailAndOtpCodeAndVerifiedFalse(email, otpCode);
            
            if (otpOptional.isEmpty()) {
                // Check if OTP was already used
                otpOptional = otpRepository.findFirstByEmailAndVerifiedFalseOrderByCreatedAtDesc(email);
                if (otpOptional.isEmpty() || !otpOptional.get().getOtpCode().equals(otpCode)) {
                    return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                        .body(new ApiResponse<>(false, "Invalid or already used OTP", null));
                }
            }
            
            OTP otp = otpOptional.get();
            
            // Check if OTP is expired
            if (otp.isExpired()) {
                return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(new ApiResponse<>(false, "OTP has expired. Please request a new one", null));
            }
            
            // Find user
            Optional<User> userOptional = userRepository.findByEmail(email);
            if (userOptional.isEmpty()) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(new ApiResponse<>(false, "User not found", null));
            }
            
            User user = userOptional.get();
            
            // Update password
            user.setPassword(passwordEncoder.encode(newPassword));
            userRepository.save(user);
            
            // Mark OTP as verified (used)
            otp.setVerified(true);
            otpRepository.save(otp);
            
            // Send confirmation email
            emailService.sendPasswordResetConfirmation(email);
            
            logger.info("Password reset successfully for user: {}", user.getUsername());
            
            return ResponseEntity.ok(new ApiResponse<>(true, 
                "Password reset successfully. You can now login with your new password", null));
                
        } catch (Exception e) {
            logger.error("Error resetting password: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiResponse<>(false, "Failed to reset password: " + e.getMessage(), null));
        }
    }
}
