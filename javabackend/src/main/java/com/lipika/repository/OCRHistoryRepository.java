package com.lipika.repository;

import com.lipika.dto.OCRHistoryDTO;
import com.lipika.model.OCRHistory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface OCRHistoryRepository extends JpaRepository<OCRHistory, Long> {
    
    // Find by recognized text containing search term
    Page<OCRHistory> findByRecognizedTextContainingIgnoreCase(String search, Pageable pageable);
    
    // Find by confidence range
    Page<OCRHistory> findByConfidenceBetween(Double minConfidence, Double maxConfidence, Pageable pageable);
    
    // Find by timestamp range
    Page<OCRHistory> findByTimestampBetween(LocalDateTime startDate, LocalDateTime endDate, Pageable pageable);
    
    // Find by text containing and confidence range
    Page<OCRHistory> findByRecognizedTextContainingIgnoreCaseAndConfidenceBetween(
            String search, Double minConfidence, Double maxConfidence, Pageable pageable);
    
    // Find by text containing and date range
    Page<OCRHistory> findByRecognizedTextContainingIgnoreCaseAndTimestampBetween(
            String search, LocalDateTime startDate, LocalDateTime endDate, Pageable pageable);
    
    // Find by confidence and date range
    Page<OCRHistory> findByConfidenceBetweenAndTimestampBetween(
            Double minConfidence, Double maxConfidence, 
            LocalDateTime startDate, LocalDateTime endDate, Pageable pageable);
    
    // Find by all filters
    Page<OCRHistory> findByRecognizedTextContainingIgnoreCaseAndConfidenceBetweenAndTimestampBetween(
            String search, Double minConfidence, Double maxConfidence,
            LocalDateTime startDate, LocalDateTime endDate, Pageable pageable);
    
    // Find records after a specific date
    List<OCRHistory> findByTimestampAfter(LocalDateTime date);
    
    // Find records between dates
    List<OCRHistory> findByTimestampBetween(LocalDateTime startDate, LocalDateTime endDate);
    
    // Calculate average confidence
    @Query("SELECT AVG(h.confidence) FROM OCRHistory h WHERE h.confidence IS NOT NULL")
    Double findAverageConfidence();
    
    // Count total characters
    @Query("SELECT COALESCE(SUM(h.characterCount), 0) FROM OCRHistory h WHERE h.characterCount IS NOT NULL")
    Long findTotalCharacterCount();
    
    // Find by user ID
    Page<OCRHistory> findByUserId(Long userId, Pageable pageable);
    
    // Count by registered status
    long countByIsRegistered(Boolean isRegistered);
    
    // Find by registered status
    Page<OCRHistory> findByIsRegistered(Boolean isRegistered, Pageable pageable);
    
    // Get OCR History with User information
    @Query("SELECT new com.lipika.dto.OCRHistoryDTO(" +
           "h.id, h.userId, u.username, u.email, u.role, " +
           "h.isRegistered, h.ipAddress, h.cookieId, " +
           "h.imageFilename, h.imagePath, h.recognizedText, " +
           "h.characterCount, h.confidence, h.timestamp, h.language) " +
           "FROM OCRHistory h LEFT JOIN h.user u")
    Page<OCRHistoryDTO> findAllWithUserInfo(Pageable pageable);
    
    // Get OCR History with User information and search filter
    @Query("SELECT new com.lipika.dto.OCRHistoryDTO(" +
           "h.id, h.userId, u.username, u.email, u.role, " +
           "h.isRegistered, h.ipAddress, h.cookieId, " +
           "h.imageFilename, h.imagePath, h.recognizedText, " +
           "h.characterCount, h.confidence, h.timestamp, h.language) " +
           "FROM OCRHistory h LEFT JOIN h.user u " +
           "WHERE LOWER(h.recognizedText) LIKE LOWER(CONCAT('%', :search, '%'))")
    Page<OCRHistoryDTO> findAllWithUserInfoAndSearch(@Param("search") String search, Pageable pageable);
    
    // Get OCR History with User information and date range
    @Query("SELECT new com.lipika.dto.OCRHistoryDTO(" +
           "h.id, h.userId, u.username, u.email, u.role, " +
           "h.isRegistered, h.ipAddress, h.cookieId, " +
           "h.imageFilename, h.imagePath, h.recognizedText, " +
           "h.characterCount, h.confidence, h.timestamp, h.language) " +
           "FROM OCRHistory h LEFT JOIN h.user u " +
           "WHERE h.timestamp BETWEEN :startDate AND :endDate")
    Page<OCRHistoryDTO> findAllWithUserInfoAndDateRange(
        @Param("startDate") LocalDateTime startDate, 
        @Param("endDate") LocalDateTime endDate, 
        Pageable pageable);
    
    // Get OCR History with User information, search and date range
    @Query("SELECT new com.lipika.dto.OCRHistoryDTO(" +
           "h.id, h.userId, u.username, u.email, u.role, " +
           "h.isRegistered, h.ipAddress, h.cookieId, " +
           "h.imageFilename, h.imagePath, h.recognizedText, " +
           "h.characterCount, h.confidence, h.timestamp, h.language) " +
           "FROM OCRHistory h LEFT JOIN h.user u " +
           "WHERE LOWER(h.recognizedText) LIKE LOWER(CONCAT('%', :search, '%')) " +
           "AND h.timestamp BETWEEN :startDate AND :endDate")
    Page<OCRHistoryDTO> findAllWithUserInfoSearchAndDateRange(
        @Param("search") String search,
        @Param("startDate") LocalDateTime startDate, 
        @Param("endDate") LocalDateTime endDate, 
        Pageable pageable);
}


