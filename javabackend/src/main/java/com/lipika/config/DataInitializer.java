package com.lipika.config;

import com.lipika.model.User;
import com.lipika.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * Data Initializer Component
 * Creates default admin user on application startup if it doesn't exist
 */
@Component
public class DataInitializer implements CommandLineRunner {
    
    private static final Logger log = LoggerFactory.getLogger(DataInitializer.class);
    
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    
    public DataInitializer(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }
    
    @Value("${admin.default.username:admin}")
    private String adminUsername;
    
    @Value("${admin.default.password:admin123}")
    private String adminPassword;
    
    @Value("${admin.default.email:admin@lipika.com}")
    private String adminEmail;
    
    @Override
    public void run(String... args) {
        initializeAdminUser();
    }
    
    private void initializeAdminUser() {
        // Check if admin user already exists
        if (userRepository.existsByUsername(adminUsername)) {
            log.info("Admin user '{}' already exists. Skipping initialization.", adminUsername);
            return;
        }
        
        // Check if admin email already exists
        if (userRepository.existsByEmail(adminEmail)) {
            log.warn("Admin email '{}' already exists with different username. Skipping initialization.", adminEmail);
            return;
        }
        
        // Create admin user
        User admin = new User();
        admin.setUsername(adminUsername);
        admin.setEmail(adminEmail);
        admin.setPasswordHash(passwordEncoder.encode(adminPassword));
        admin.setRole("ADMIN");
        admin.setIsActive(true);
        admin.setCreatedAt(LocalDateTime.now());
        admin.setUpdatedAt(LocalDateTime.now());
        
        admin = userRepository.save(admin);
        
        log.info("========================================");
        log.info("Default Admin User Created Successfully");
        log.info("========================================");
        log.info("Username: {}", adminUsername);
        log.info("Email: {}", adminEmail);
        log.info("Password: {} (Please change this after first login)", adminPassword);
        log.info("Role: ADMIN");
        log.info("========================================");
    }
}

