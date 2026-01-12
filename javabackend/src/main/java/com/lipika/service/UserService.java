package com.lipika.service;

import com.lipika.dto.*;
import com.lipika.model.PasswordResetToken;
import com.lipika.model.User;
import com.lipika.repository.PasswordResetTokenRepository;
import com.lipika.repository.UserRepository;
import com.lipika.util.JwtUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.UUID;

@Service
public class UserService {
    
    private static final Logger log = LoggerFactory.getLogger(UserService.class);
    
    private final UserRepository userRepository;
    private final PasswordResetTokenRepository resetTokenRepository;
    private final PasswordEncoder passwordEncoder;
    private final EmailService emailService;
    private final JwtUtil jwtUtil;
    
    public UserService(UserRepository userRepository,
                      PasswordResetTokenRepository resetTokenRepository,
                      PasswordEncoder passwordEncoder,
                      EmailService emailService,
                      JwtUtil jwtUtil) {
        this.userRepository = userRepository;
        this.resetTokenRepository = resetTokenRepository;
        this.passwordEncoder = passwordEncoder;
        this.emailService = emailService;
        this.jwtUtil = jwtUtil;
    }
    
    @Transactional
    public AuthResponse register(RegisterRequest request) {
        // Check if username exists
        if (userRepository.findByUsername(request.getUsername()).isPresent()) {
            throw new RuntimeException("Username already exists");
        }
        
        // Check if email exists
        if (userRepository.findByEmail(request.getEmail()).isPresent()) {
            throw new RuntimeException("Email already exists");
        }
        
        // Create new user
        User user = new User();
        user.setUsername(request.getUsername());
        user.setEmail(request.getEmail());
        user.setPasswordHash(passwordEncoder.encode(request.getPassword()));
        user.setRole("USER");
        user.setIsActive(true);
        user.setUsageCount(0);
        user.setUsageLimit(10);
        user.setIsPremium(false);
        
        user = userRepository.save(user);
        log.info("New user registered: {}", user.getUsername());
        
        // Send welcome email (asynchronous, won't fail registration if email fails)
        try {
            emailService.sendWelcomeEmail(user.getEmail(), user.getUsername());
        } catch (Exception e) {
            log.warn("Failed to send welcome email, but registration succeeded", e);
        }
        
        // Generate token
        String token = jwtUtil.generateToken(user.getId(), user.getUsername(), user.getEmail(), user.getRole());
        
        return new AuthResponse(
            token,
            "Bearer",
            user.getId(),
            user.getUsername(),
            user.getEmail(),
            user.getRole(),
            user.getUsageLimit() - user.getUsageCount()
        );
    }
    
    @Transactional
    public AuthResponse login(LoginRequest request) {
        // Find user by username or email
        Optional<User> userOpt = userRepository.findByUsername(request.getUsernameOrEmail())
            .or(() -> userRepository.findByEmail(request.getUsernameOrEmail()));
        
        if (userOpt.isEmpty()) {
            throw new RuntimeException("Invalid username/email or password");
        }
        
        User user = userOpt.get();
        
        // Check password
        if (!passwordEncoder.matches(request.getPassword(), user.getPasswordHash())) {
            throw new RuntimeException("Invalid username/email or password");
        }
        
        // Check if user is active
        if (!user.getIsActive()) {
            throw new RuntimeException("Account is deactivated");
        }
        
        // Update last login
        user.setLastLogin(LocalDateTime.now());
        userRepository.save(user);
        
        log.info("User logged in: {}", user.getUsername());
        
        // Calculate remaining scans
        Integer remainingScans = user.getIsPremium() ? null : user.getUsageLimit() - user.getUsageCount();
        
        // Generate token
        String token = jwtUtil.generateToken(user.getId(), user.getUsername(), user.getEmail(), user.getRole());
        
        return new AuthResponse(
            token,
            "Bearer",
            user.getId(),
            user.getUsername(),
            user.getEmail(),
            user.getRole(),
            remainingScans
        );
    }
    
    @Transactional
    public void forgotPassword(ForgotPasswordRequest request) {
        Optional<User> userOpt = userRepository.findByEmail(request.getEmail());
        
        if (userOpt.isEmpty()) {
            // Don't reveal if email exists or not (security best practice)
            log.info("Password reset requested for non-existent email: {}", request.getEmail());
            return;
        }
        
        User user = userOpt.get();
        
        // Delete any existing unused tokens for this user
        resetTokenRepository.deleteByUser(user);
        
        // Generate reset token
        String token = UUID.randomUUID().toString();
        LocalDateTime expiryDate = LocalDateTime.now().plusHours(1);
        
        PasswordResetToken resetToken = new PasswordResetToken(token, user, expiryDate);
        resetTokenRepository.save(resetToken);
        
        // Send password reset email
        emailService.sendPasswordResetEmail(user.getEmail(), user.getUsername(), token);
        
        log.info("Password reset token generated for user: {}", user.getUsername());
    }
    
    @Transactional
    public void resetPassword(ResetPasswordRequest request) {
        Optional<PasswordResetToken> tokenOpt = resetTokenRepository.findByToken(request.getToken());
        
        if (tokenOpt.isEmpty()) {
            throw new RuntimeException("Invalid reset token");
        }
        
        PasswordResetToken resetToken = tokenOpt.get();
        
        if (resetToken.getUsed()) {
            throw new RuntimeException("Reset token already used");
        }
        
        if (resetToken.isExpired()) {
            throw new RuntimeException("Reset token expired");
        }
        
        // Update user password
        User user = resetToken.getUser();
        user.setPasswordHash(passwordEncoder.encode(request.getNewPassword()));
        userRepository.save(user);
        
        // Mark token as used
        resetToken.setUsed(true);
        resetTokenRepository.save(resetToken);
        
        log.info("Password reset successful for user: {}", user.getUsername());
    }
    
    public UserProfileResponse getUserProfile(Long userId) {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new RuntimeException("User not found"));
        
        return new UserProfileResponse(
            user.getId(),
            user.getUsername(),
            user.getEmail(),
            user.getRole(),
            user.getUsageCount(),
            user.getUsageLimit(),
            user.getIsPremium(),
            user.getPremiumUntil(),
            user.getCreatedAt(),
            user.getLastLogin()
        );
    }
    
    @Transactional
    public void incrementUsageCount(Long userId) {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new RuntimeException("User not found"));
        
        // Don't increment if user is premium
        if (user.getIsPremium()) {
            return;
        }
        
        user.setUsageCount(user.getUsageCount() + 1);
        userRepository.save(user);
        
        log.info("Usage count incremented for user: {} (count: {})", user.getUsername(), user.getUsageCount());
    }
    
    public boolean hasReachedLimit(Long userId) {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new RuntimeException("User not found"));
        
        // Premium users have no limit
        if (user.getIsPremium()) {
            return false;
        }
        
        return user.getUsageCount() >= user.getUsageLimit();
    }
    
    @Transactional
    public void changePassword(Long userId, String currentPassword, String newPassword) {
        User user = userRepository.findById(userId)
            .orElseThrow(() -> new RuntimeException("User not found"));
        
        // Verify current password
        if (!passwordEncoder.matches(currentPassword, user.getPasswordHash())) {
            throw new RuntimeException("Current password is incorrect");
        }
        
        // Check if new password is same as current
        if (passwordEncoder.matches(newPassword, user.getPasswordHash())) {
            throw new RuntimeException("New password must be different from current password");
        }
        
        // Update password
        user.setPasswordHash(passwordEncoder.encode(newPassword));
        userRepository.save(user);
        
        log.info("Password changed successfully for user: {}", user.getUsername());
    }
}

