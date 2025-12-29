package com.lipika.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

/**
 * Database Initialization Service
 * Handles database schema creation and initialization
 * Replaces the need for manual schema.sql execution
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class DatabaseInitializationService {
    
    private final JdbcTemplate jdbcTemplate;
    
    @Value("${spring.jpa.hibernate.ddl-auto:validate}")
    private String ddlAuto;
    
    @PostConstruct
    public void initializeDatabase() {
        // Only initialize if not using Hibernate auto-ddl
        if ("validate".equals(ddlAuto) || "none".equals(ddlAuto)) {
            log.info("Initializing database schema...");
            try {
                executeSchemaScript();
                log.info("Database schema initialized successfully");
            } catch (Exception e) {
                log.error("Error initializing database schema", e);
            }
        } else {
            log.info("Skipping manual schema initialization - using Hibernate ddl-auto: {}", ddlAuto);
        }
    }
    
    private void executeSchemaScript() {
        try {
            // Create database if not exists
            jdbcTemplate.execute("CREATE DATABASE IF NOT EXISTS lipika CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci");
            jdbcTemplate.execute("USE lipika");
            
            // OCR History Table
            jdbcTemplate.execute("""
                CREATE TABLE IF NOT EXISTS ocr_history (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id BIGINT NULL,
                    is_registered BOOLEAN DEFAULT FALSE,
                    ip_address VARCHAR(45),
                    cookie_id VARCHAR(255),
                    image_filename VARCHAR(500),
                    recognized_text TEXT,
                    character_count INT,
                    confidence DOUBLE,
                    timestamp DATETIME NOT NULL,
                    language VARCHAR(50) DEFAULT 'devanagari',
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_confidence (confidence),
                    INDEX idx_language (language),
                    INDEX idx_user_id (user_id),
                    INDEX idx_is_registered (is_registered),
                    INDEX idx_ip_address (ip_address),
                    FULLTEXT INDEX ft_recognized_text (recognized_text),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """);
            
            // System Settings Table
            jdbcTemplate.execute("""
                CREATE TABLE IF NOT EXISTS system_settings (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    setting_key VARCHAR(100) UNIQUE NOT NULL,
                    setting_value TEXT,
                    description VARCHAR(500),
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_setting_key (setting_key)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """);
            
            // Insert default settings
            jdbcTemplate.update("""
                INSERT INTO system_settings (setting_key, setting_value, description) VALUES
                ('ocr_service_url', 'http://localhost:5000', 'Python OCR service URL'),
                ('translation_api_enabled', 'true', 'Enable/disable translation API'),
                ('translation_api_url', 'https://libretranslate.de/translate', 'Translation API URL'),
                ('max_file_size', '10485760', 'Maximum file upload size in bytes (10MB)'),
                ('supported_formats', 'image/jpeg,image/png,image/jpg,image/webp', 'Supported image formats')
                ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value)
                """);
            
            // Users Table with Roles
            jdbcTemplate.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL DEFAULT 'USER',
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    last_login DATETIME,
                    INDEX idx_username (username),
                    INDEX idx_email (email),
                    INDEX idx_role (role),
                    INDEX idx_is_active (is_active)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """);
            
            // Trial Tracking Table
            jdbcTemplate.execute("""
                CREATE TABLE IF NOT EXISTS trial_tracking (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    ip_address VARCHAR(45) NOT NULL,
                    cookie_id VARCHAR(255),
                    fingerprint VARCHAR(255),
                    trial_count INT DEFAULT 0,
                    first_attempt DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_attempt DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    is_blocked BOOLEAN DEFAULT FALSE,
                    INDEX idx_ip_address (ip_address),
                    INDEX idx_cookie_id (cookie_id),
                    INDEX idx_fingerprint (fingerprint),
                    INDEX idx_is_blocked (is_blocked),
                    UNIQUE KEY unique_tracking (ip_address, cookie_id, fingerprint)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """);
            
            log.info("All database tables created successfully");
            
        } catch (Exception e) {
            log.error("Error executing schema script", e);
            throw new RuntimeException("Failed to initialize database schema", e);
        }
    }
}

