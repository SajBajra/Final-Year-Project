package com.lipika.service;

import com.lipika.model.OCRHistory;

import java.util.List;
import java.util.Map;

public interface AdminService {
    
    /**
     * Get dashboard statistics
     */
    Map<String, Object> getDashboardStats();
    
    /**
     * Get OCR history with pagination
     */
    Map<String, Object> getOCRHistory(int page, int size);
    
    /**
     * Get OCR history by ID
     */
    OCRHistory getOCRHistoryById(Long id);
    
    /**
     * Delete OCR history by ID
     */
    boolean deleteOCRHistory(Long id);
    
    /**
     * Get system settings
     */
    Map<String, Object> getSettings();
    
    /**
     * Update system settings
     */
    boolean updateSettings(Map<String, Object> settings);
    
    /**
     * Save OCR history (called by OCR service)
     */
    void saveOCRHistory(String imageFilename, String recognizedText, 
                       Integer characterCount, Double confidence);
}

