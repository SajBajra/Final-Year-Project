-- Lipika OCR System Database Schema
-- MySQL Database Setup Script (XAMPP)

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS lipika CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE lipika;

-- OCR History Table (updated with user tracking)
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- System Settings Table (for future use)
CREATE TABLE IF NOT EXISTS system_settings (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value TEXT,
    description VARCHAR(500),
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_setting_key (setting_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert default settings
INSERT INTO system_settings (setting_key, setting_value, description) VALUES
('ocr_service_url', 'http://localhost:5000', 'Python OCR service URL'),
('translation_api_enabled', 'true', 'Enable/disable translation API'),
('translation_api_url', 'https://libretranslate.de/translate', 'Translation API URL'),
('max_file_size', '10485760', 'Maximum file upload size in bytes (10MB)'),
('supported_formats', 'image/jpeg,image/png,image/jpg,image/webp', 'Supported image formats')
ON DUPLICATE KEY UPDATE setting_value = VALUES(setting_value);

-- Users Table with Roles
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Note: Default admin user is created automatically by DataInitializer component on application startup
-- Default credentials (configured in application.properties):
-- Username: admin
-- Password: admin123
-- Email: admin@lipika.com
-- Role: ADMIN
-- 
-- IMPORTANT: Change the default admin password after first login!

-- Trial Tracking Table (for unregistered users)
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create indexes for better query performance
-- Index on timestamp for analytics queries
-- Index on confidence for filtering
-- Fulltext index on recognized_text for search (already created above)

