-- Lipika OCR System Database Schema
-- MySQL Database Setup Script

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS lipika_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE lipika_db;

-- OCR History Table
CREATE TABLE IF NOT EXISTS ocr_history (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_filename VARCHAR(500),
    recognized_text TEXT,
    character_count INT,
    confidence DOUBLE,
    timestamp DATETIME NOT NULL,
    language VARCHAR(50) DEFAULT 'devanagari',
    INDEX idx_timestamp (timestamp),
    INDEX idx_confidence (confidence),
    INDEX idx_language (language),
    FULLTEXT INDEX ft_recognized_text (recognized_text)
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

-- Admin Users Table (for future use - currently password is in-memory)
CREATE TABLE IF NOT EXISTS admin_users (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_username (username),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Create indexes for better query performance
-- Index on timestamp for analytics queries
-- Index on confidence for filtering
-- Fulltext index on recognized_text for search (already created above)

