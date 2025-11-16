package com.lipika.service;

import com.lipika.model.OCRHistory;

import java.time.LocalDateTime;
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
     * Get OCR history with search, filters, and sorting
     * @param page Page number (0-based)
     * @param size Page size
     * @param search Search term for recognized text
     * @param minConfidence Minimum confidence threshold
     * @param maxConfidence Maximum confidence threshold
     * @param startDate Start date filter (optional)
     * @param endDate End date filter (optional)
     * @param sortBy Sort field (timestamp, confidence, characterCount)
     * @param sortOrder Sort order (asc, desc)
     * @return Paginated and filtered OCR history
     */
    Map<String, Object> getOCRHistoryFiltered(int page, int size, String search, 
                                              Double minConfidence, Double maxConfidence,
                                              LocalDateTime startDate, LocalDateTime endDate,
                                              String sortBy, String sortOrder);
    
    /**
     * Get OCR history by ID
     */
    OCRHistory getOCRHistoryById(Long id);
    
    /**
     * Delete OCR history by ID
     */
    boolean deleteOCRHistory(Long id);
    
    /**
     * Bulk delete OCR history by IDs
     */
    boolean bulkDeleteOCRHistory(List<Long> ids);
    
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
                       Integer characterCount, Double confidence,
                       Long userId, Boolean isRegistered, String ipAddress, String cookieId);
    
    /**
     * Get analytics data for charts
     * @param period Period type (daily, weekly, monthly)
     * @param days Number of days to look back
     * @return Analytics data with time series
     */
    Map<String, Object> getAnalytics(String period, int days);
    
    /**
     * Get character statistics
     * @return Character frequency and accuracy statistics
     */
    Map<String, Object> getCharacterStatistics();
    
    /**
     * Export OCR history to CSV format
     * @param search Search filter
     * @param minConfidence Minimum confidence
     * @param maxConfidence Maximum confidence
     * @param startDate Start date
     * @param endDate End date
     * @return CSV content as string
     */
    String exportOCRHistoryToCSV(String search, Double minConfidence, Double maxConfidence,
                                 LocalDateTime startDate, LocalDateTime endDate);
    
    /**
     * Change admin password
     * @param currentPassword Current password
     * @param newPassword New password
     * @return true if password changed successfully, false otherwise
     */
    boolean changePassword(String currentPassword, String newPassword);
    
    /**
     * Get total record count for diagnostics
     */
    long getTotalRecordCount();
    
    /**
     * Get a sample record for diagnostics
     */
    Map<String, Object> getSampleRecord();
}

