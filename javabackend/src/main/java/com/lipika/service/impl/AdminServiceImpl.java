package com.lipika.service.impl;

import com.lipika.model.OCRHistory;
import com.lipika.service.AdminService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Slf4j
@Service
public class AdminServiceImpl implements AdminService {
    
    // In-memory storage for OCR history (replace with database in production)
    private final Map<Long, OCRHistory> ocrHistoryStore = new ConcurrentHashMap<>();
    private final AtomicLong idGenerator = new AtomicLong(1);
    
    // System settings
    private final Map<String, Object> settings = new ConcurrentHashMap<>();
    
    public AdminServiceImpl() {
        // Initialize default settings
        settings.put("ocrServiceUrl", "http://localhost:5000");
        settings.put("translationApiEnabled", true);
        settings.put("translationApiUrl", "https://libretranslate.de/translate");
        settings.put("maxFileSize", 10485760); // 10MB
        settings.put("supportedFormats", Arrays.asList("image/jpeg", "image/png", "image/jpg", "image/webp"));
    }
    
    /**
     * Save OCR history (called by OCR service)
     */
    @Override
    public void saveOCRHistory(String imageFilename, String recognizedText, 
                               Integer characterCount, Double confidence) {
        OCRHistory history = new OCRHistory();
        history.setId(idGenerator.getAndIncrement());
        history.setImageFilename(imageFilename);
        history.setRecognizedText(recognizedText);
        history.setCharacterCount(characterCount);
        history.setConfidence(confidence);
        history.setTimestamp(LocalDateTime.now());
        history.setLanguage("devanagari");
        
        ocrHistoryStore.put(history.getId(), history);
        log.info("Saved OCR history: ID={}, Text={}", history.getId(), recognizedText);
    }
    
    @Override
    public Map<String, Object> getDashboardStats() {
        Map<String, Object> stats = new HashMap<>();
        
        int totalRecords = ocrHistoryStore.size();
        double avgConfidence = ocrHistoryStore.values().stream()
                .mapToDouble(h -> h.getConfidence() != null ? h.getConfidence() : 0.0)
                .average()
                .orElse(0.0);
        
        int totalCharacters = ocrHistoryStore.values().stream()
                .mapToInt(h -> h.getCharacterCount() != null ? h.getCharacterCount() : 0)
                .sum();
        
        // Recent activity (last 24 hours)
        LocalDateTime yesterday = LocalDateTime.now().minusDays(1);
        long recentActivity = ocrHistoryStore.values().stream()
                .filter(h -> h.getTimestamp().isAfter(yesterday))
                .count();
        
        stats.put("totalRecords", totalRecords);
        stats.put("avgConfidence", Math.round(avgConfidence * 100.0) / 100.0);
        stats.put("totalCharacters", totalCharacters);
        stats.put("recentActivity", recentActivity);
        stats.put("timestamp", LocalDateTime.now().toString());
        
        return stats;
    }
    
    @Override
    public Map<String, Object> getOCRHistory(int page, int size) {
        List<OCRHistory> allHistory = new ArrayList<>(ocrHistoryStore.values());
        
        // Sort by timestamp descending (newest first)
        allHistory.sort((a, b) -> b.getTimestamp().compareTo(a.getTimestamp()));
        
        // Pagination
        int start = page * size;
        int end = Math.min(start + size, allHistory.size());
        
        List<OCRHistory> pageData = start < allHistory.size() 
                ? allHistory.subList(start, end) 
                : new ArrayList<>();
        
        Map<String, Object> result = new HashMap<>();
        result.put("data", pageData);
        result.put("page", page);
        result.put("size", size);
        result.put("total", allHistory.size());
        result.put("totalPages", (int) Math.ceil((double) allHistory.size() / size));
        
        return result;
    }
    
    @Override
    public OCRHistory getOCRHistoryById(Long id) {
        return ocrHistoryStore.get(id);
    }
    
    @Override
    public boolean deleteOCRHistory(Long id) {
        OCRHistory removed = ocrHistoryStore.remove(id);
        if (removed != null) {
            log.info("Deleted OCR history: ID={}", id);
            return true;
        }
        return false;
    }
    
    @Override
    public Map<String, Object> getSettings() {
        return new HashMap<>(settings);
    }
    
    @Override
    public boolean updateSettings(Map<String, Object> newSettings) {
        try {
            settings.putAll(newSettings);
            log.info("Updated settings: {}", newSettings);
            return true;
        } catch (Exception e) {
            log.error("Error updating settings", e);
            return false;
        }
    }
}

