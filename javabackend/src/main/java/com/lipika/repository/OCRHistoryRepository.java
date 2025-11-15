package com.lipika.repository;

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
}


