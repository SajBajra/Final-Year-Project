package com.lipika.service;

import com.lipika.dto.AuthResponse;
import com.lipika.dto.LoginRequest;
import com.lipika.model.User;
import com.lipika.repository.UserRepository;
import com.lipika.util.JwtUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;

@Service
public class AuthService {
    
    private static final Logger log = LoggerFactory.getLogger(AuthService.class);
    
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;
    private final TrialTrackingService trialTrackingService;
    
    public AuthService(UserRepository userRepository, PasswordEncoder passwordEncoder, 
                      JwtUtil jwtUtil, TrialTrackingService trialTrackingService) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtUtil = jwtUtil;
        this.trialTrackingService = trialTrackingService;
    }
    
    public AuthResponse login(LoginRequest request) {
        // Find user by username or email
        Optional<User> userOpt = userRepository.findByUsername(request.getUsernameOrEmail())
            .or(() -> userRepository.findByEmail(request.getUsernameOrEmail()));
        
        if (userOpt.isEmpty()) {
            throw new RuntimeException("Invalid username/email or password");
        }
        
        User user = userOpt.get();
        
        // Ensure only ADMIN users can log in via this endpoint.
        if (!"ADMIN".equalsIgnoreCase(user.getRole())) {
            throw new RuntimeException("Only admin accounts can log in here");
        }
        
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
        
        // Generate token
        String token = jwtUtil.generateToken(user.getId(), user.getUsername(), user.getEmail(), user.getRole());
        
        return new AuthResponse(
            token,
            "Bearer",
            user.getId(),
            user.getUsername(),
            user.getEmail(),
            user.getRole(),
            null, // Unlimited for admin users
            true, // Admins are always premium
            null  // No expiry for admins
        );
    }
}

