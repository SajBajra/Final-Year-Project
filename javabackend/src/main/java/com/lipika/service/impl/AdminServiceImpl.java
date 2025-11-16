package com.lipika.service.impl;

import com.lipika.model.OCRHistory;
import com.lipika.repository.OCRHistoryRepository;
import com.lipika.service.AdminService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class AdminServiceImpl implements AdminService {
    
    private final OCRHistoryRepository ocrHistoryRepository;
    
    // System settings (still in-memory - can be moved to DB later)
    private final Map<String, Object> settings = new ConcurrentHashMap<>();
    
    // Admin password (in production, use proper password hashing and database)
    private String adminPassword = "admin"; // Default password
    
    {
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
    @Transactional
    public void saveOCRHistory(String imageFilename, String recognizedText, 
                               Integer characterCount, Double confidence,
                               Long userId, Boolean isRegistered, String ipAddress, String cookieId) {
        OCRHistory history = new OCRHistory();
        history.setImageFilename(imageFilename);
        history.setRecognizedText(recognizedText);
        history.setCharacterCount(characterCount);
        history.setConfidence(confidence);
        history.setTimestamp(LocalDateTime.now());
        history.setLanguage("devanagari");
        history.setUserId(userId);
        history.setIsRegistered(isRegistered != null ? isRegistered : false);
        history.setIpAddress(ipAddress);
        history.setCookieId(cookieId);
        
        OCRHistory saved = ocrHistoryRepository.save(history);
        log.info("Saved OCR history to database: ID={}, UserId={}, IsRegistered={}, Text={}", 
                saved.getId(), userId, isRegistered, recognizedText);
    }
    
    @Override
    public Map<String, Object> getDashboardStats() {
        Map<String, Object> stats = new HashMap<>();
        
        long totalRecords = ocrHistoryRepository.count();
        
        Long totalCharsObj = ocrHistoryRepository.findTotalCharacterCount();
        int totalCharacters = totalCharsObj != null ? totalCharsObj.intValue() : 0;
        
        // Recent activity (last 24 hours)
        LocalDateTime yesterday = LocalDateTime.now().minusDays(1);
        long recentActivity = ocrHistoryRepository.findByTimestampAfter(yesterday).size();
        
        // User type distribution
        long registeredCount = ocrHistoryRepository.countByIsRegistered(true);
        long unregisteredCount = ocrHistoryRepository.countByIsRegistered(false);
        
        stats.put("totalRecords", totalRecords);
        stats.put("totalCharacters", totalCharacters);
        stats.put("recentActivity", recentActivity);
        stats.put("registeredCount", registeredCount);
        stats.put("unregisteredCount", unregisteredCount);
        stats.put("timestamp", LocalDateTime.now().toString());
        
        return stats;
    }
    
    @Override
    public Map<String, Object> getOCRHistory(int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "timestamp"));
        Page<OCRHistory> historyPage = ocrHistoryRepository.findAll(pageable);
        
        log.info("getOCRHistory: page={}, size={}, totalElements={}, totalPages={}, contentSize={}", 
            page, size, historyPage.getTotalElements(), historyPage.getTotalPages(), historyPage.getContent().size());
        
        Map<String, Object> result = new HashMap<>();
        result.put("data", historyPage.getContent());
        result.put("page", page);
        result.put("size", size);
        result.put("total", historyPage.getTotalElements());
        result.put("totalPages", historyPage.getTotalPages());
        
        return result;
    }
    
    @Override
    public OCRHistory getOCRHistoryById(Long id) {
        return ocrHistoryRepository.findById(id).orElse(null);
    }
    
    @Override
    @Transactional
    public boolean deleteOCRHistory(Long id) {
        if (ocrHistoryRepository.existsById(id)) {
            ocrHistoryRepository.deleteById(id);
            log.info("Deleted OCR history: ID={}", id);
            return true;
        }
        return false;
    }
    
    @Override
    @Transactional
    public boolean bulkDeleteOCRHistory(List<Long> ids) {
        int deleted = 0;
        for (Long id : ids) {
            if (ocrHistoryRepository.existsById(id)) {
                ocrHistoryRepository.deleteById(id);
                deleted++;
            }
        }
        log.info("Bulk deleted {} OCR history records", deleted);
        return deleted > 0;
    }
    
    @Override
    public Map<String, Object> getOCRHistoryFiltered(int page, int size, String search,
                                                      Double minConfidence, Double maxConfidence,
                                                      LocalDateTime startDate, LocalDateTime endDate,
                                                      String sortBy, String sortOrder) {
        // Build sort
        Sort.Direction direction = "asc".equalsIgnoreCase(sortOrder) 
                ? Sort.Direction.ASC : Sort.Direction.DESC;
        
        String sortField = sortBy != null ? sortBy : "timestamp";
        // Map sort field names
        if ("charactercount".equalsIgnoreCase(sortField)) {
            sortField = "characterCount";
        }
        
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortField));
        
        Page<OCRHistory> historyPage;
        
        // Build query based on provided filters
        if (search != null && !search.trim().isEmpty() && 
            minConfidence != null && maxConfidence != null &&
            startDate != null && endDate != null) {
            // All filters
            historyPage = ocrHistoryRepository.findByRecognizedTextContainingIgnoreCaseAndConfidenceBetweenAndTimestampBetween(
                    search, minConfidence, maxConfidence, startDate, endDate, pageable);
        } else if (search != null && !search.trim().isEmpty() && 
                   minConfidence != null && maxConfidence != null) {
            // Text + confidence
            historyPage = ocrHistoryRepository.findByRecognizedTextContainingIgnoreCaseAndConfidenceBetween(
                    search, minConfidence, maxConfidence, pageable);
        } else if (search != null && !search.trim().isEmpty() && 
                   startDate != null && endDate != null) {
            // Text + date
            historyPage = ocrHistoryRepository.findByRecognizedTextContainingIgnoreCaseAndTimestampBetween(
                    search, startDate, endDate, pageable);
        } else if (minConfidence != null && maxConfidence != null && 
                   startDate != null && endDate != null) {
            // Confidence + date
            historyPage = ocrHistoryRepository.findByConfidenceBetweenAndTimestampBetween(
                    minConfidence, maxConfidence, startDate, endDate, pageable);
        } else if (search != null && !search.trim().isEmpty()) {
            // Text only
            historyPage = ocrHistoryRepository.findByRecognizedTextContainingIgnoreCase(search, pageable);
        } else if (minConfidence != null && maxConfidence != null) {
            // Confidence only
            historyPage = ocrHistoryRepository.findByConfidenceBetween(minConfidence, maxConfidence, pageable);
        } else if (startDate != null && endDate != null) {
            // Date only
            historyPage = ocrHistoryRepository.findByTimestampBetween(startDate, endDate, pageable);
        } else {
            // No filters
            historyPage = ocrHistoryRepository.findAll(pageable);
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("data", historyPage.getContent());
        result.put("page", page);
        result.put("size", size);
        result.put("total", historyPage.getTotalElements());
        result.put("totalPages", historyPage.getTotalPages());
        
        return result;
    }
    
    @Override
    public Map<String, Object> getAnalytics(String period, int days) {
        Map<String, Object> analytics = new HashMap<>();
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime startDate = now.minusDays(days);
        
        // Filter records within date range
        List<OCRHistory> recentHistory = ocrHistoryRepository.findByTimestampBetween(startDate, now);
        
        // Time series data (daily, weekly, monthly)
        Map<String, Integer> timeSeries = new LinkedHashMap<>();
        Map<String, Integer> characterSeries = new LinkedHashMap<>();
        
        if ("daily".equalsIgnoreCase(period)) {
            for (int i = days - 1; i >= 0; i--) {
                LocalDateTime date = now.minusDays(i);
                String key = date.toLocalDate().toString();
                
                final LocalDateTime dayStart = date.toLocalDate().atStartOfDay();
                final LocalDateTime dayEnd = date.toLocalDate().atTime(23, 59, 59);
                
                List<OCRHistory> dayRecords = recentHistory.stream()
                        .filter(h -> !h.getTimestamp().isBefore(dayStart) && !h.getTimestamp().isAfter(dayEnd))
                        .collect(Collectors.toList());
                
                timeSeries.put(key, dayRecords.size());
                characterSeries.put(key, dayRecords.stream()
                        .mapToInt(h -> h.getCharacterCount() != null ? h.getCharacterCount() : 0)
                        .sum());
            }
        } else if ("weekly".equalsIgnoreCase(period)) {
            // Group by week
            Map<String, List<OCRHistory>> weeklyGroups = recentHistory.stream()
                    .collect(Collectors.groupingBy(h -> {
                        LocalDateTime date = h.getTimestamp();
                        LocalDateTime weekStart = date.minusDays(date.getDayOfWeek().getValue() - 1);
                        return weekStart.toLocalDate().toString();
                    }));
            
            for (String week : weeklyGroups.keySet()) {
                List<OCRHistory> weekRecords = weeklyGroups.get(week);
                timeSeries.put(week, weekRecords.size());
                characterSeries.put(week, weekRecords.stream()
                        .mapToInt(h -> h.getCharacterCount() != null ? h.getCharacterCount() : 0)
                        .sum());
            }
        } else {
            // Monthly grouping
            Map<String, List<OCRHistory>> monthlyGroups = recentHistory.stream()
                    .collect(Collectors.groupingBy(h -> {
                        LocalDateTime date = h.getTimestamp();
                        return date.getYear() + "-" + String.format("%02d", date.getMonthValue());
                    }));
            
            for (String month : monthlyGroups.keySet()) {
                List<OCRHistory> monthRecords = monthlyGroups.get(month);
                timeSeries.put(month, monthRecords.size());
                characterSeries.put(month, monthRecords.stream()
                        .mapToInt(h -> h.getCharacterCount() != null ? h.getCharacterCount() : 0)
                        .sum());
            }
        }
        
        // Text length distribution (by character count)
        // Use characterCount if available, otherwise calculate from recognizedText
        Map<String, Long> textLengthDistribution = new HashMap<>();
        long shortText = recentHistory.stream()
                .filter(h -> {
                    int count = getCharacterCount(h);
                    return count > 0 && count <= 10;
                })
                .count();
        long mediumText = recentHistory.stream()
                .filter(h -> {
                    int count = getCharacterCount(h);
                    return count > 10 && count <= 50;
                })
                .count();
        long longText = recentHistory.stream()
                .filter(h -> {
                    int count = getCharacterCount(h);
                    return count > 50 && count <= 100;
                })
                .count();
        long veryLongText = recentHistory.stream()
                .filter(h -> {
                    int count = getCharacterCount(h);
                    return count > 100;
                })
                .count();
        
        textLengthDistribution.put("Short (1-10 chars)", shortText);
        textLengthDistribution.put("Medium (11-50 chars)", mediumText);
        textLengthDistribution.put("Long (51-100 chars)", longText);
        textLengthDistribution.put("Very Long (100+ chars)", veryLongText);
        
        log.info("Text length distribution: Short={}, Medium={}, Long={}, VeryLong={}, Total={}", 
            shortText, mediumText, longText, veryLongText, recentHistory.size());
        
        analytics.put("timeSeries", timeSeries);
        analytics.put("characterSeries", characterSeries);
        analytics.put("textLengthDistribution", textLengthDistribution);
        analytics.put("totalRecords", recentHistory.size());
        analytics.put("period", period);
        analytics.put("days", days);
        
        return analytics;
    }
    
    @Override
    public Map<String, Object> getCharacterStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        // Get all OCR history records
        List<OCRHistory> allHistory = ocrHistoryRepository.findAll();
        
        // Character frequency analysis
        Map<String, Long> characterFrequency = new HashMap<>();
        
        for (OCRHistory history : allHistory) {
            if (history.getRecognizedText() != null) {
                String text = history.getRecognizedText();
                
                // Count each character
                for (char c : text.toCharArray()) {
                    String charStr = String.valueOf(c);
                    characterFrequency.put(charStr, characterFrequency.getOrDefault(charStr, 0L) + 1);
                }
            }
        }
        
        // Get top 20 most frequent characters
        List<Map<String, Object>> topCharacters = characterFrequency.entrySet().stream()
                .sorted((a, b) -> Long.compare(b.getValue(), a.getValue()))
                .limit(20)
                .map(entry -> {
                    Map<String, Object> charStat = new HashMap<>();
                    charStat.put("character", entry.getKey());
                    charStat.put("frequency", entry.getValue());
                    return charStat;
                })
                .collect(Collectors.toList());
        
        stats.put("totalUniqueCharacters", characterFrequency.size());
        stats.put("topCharacters", topCharacters);
        stats.put("characterFrequency", characterFrequency);
        
        return stats;
    }
    
    @Override
    public String exportOCRHistoryToCSV(String search, LocalDateTime startDate, LocalDateTime endDate) {
        // Use the filtered method to get filtered data
        Map<String, Object> filteredResult = getOCRHistoryFiltered(
                0, Integer.MAX_VALUE, search, null, null, 
                startDate, endDate, "timestamp", "desc");
        
        @SuppressWarnings("unchecked")
        List<OCRHistory> history = (List<OCRHistory>) filteredResult.get("data");
        
        // Build CSV
        StringBuilder csv = new StringBuilder();
        csv.append("ID,Image Filename,Recognized Text,Character Count,Timestamp,Language\n");
        
        for (OCRHistory record : history) {
            csv.append(record.getId()).append(",");
            csv.append("\"").append(escapeCSV(record.getImageFilename())).append("\",");
            csv.append("\"").append(escapeCSV(record.getRecognizedText())).append("\",");
            csv.append(record.getCharacterCount() != null ? record.getCharacterCount() : 0).append(",");
            csv.append(record.getTimestamp() != null ? record.getTimestamp().toString() : "").append(",");
            csv.append(record.getLanguage() != null ? record.getLanguage() : "").append("\n");
        }
        
        return csv.toString();
    }
    
    private String escapeCSV(String value) {
        if (value == null) return "";
        return value.replace("\"", "\"\"").replace("\n", " ").replace("\r", " ");
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
    
    @Override
    public boolean changePassword(String currentPassword, String newPassword) {
        try {
            // Check if current password matches
            if (currentPassword == null || !currentPassword.equals(adminPassword)) {
                log.warn("Password change failed: Current password incorrect");
                return false;
            }
            
            // Validate new password
            if (newPassword == null || newPassword.trim().isEmpty()) {
                log.warn("Password change failed: New password cannot be empty");
                return false;
            }
            
            if (newPassword.length() < 4) {
                log.warn("Password change failed: New password must be at least 4 characters");
                return false;
            }
            
            // Update password
            adminPassword = newPassword;
            log.info("Admin password changed successfully");
            return true;
        } catch (Exception e) {
            log.error("Error changing password", e);
            return false;
        }
    }
    
    @Override
    public long getTotalRecordCount() {
        return ocrHistoryRepository.count();
    }
    
    @Override
    public Map<String, Object> getSampleRecord() {
        List<OCRHistory> records = ocrHistoryRepository.findAll(PageRequest.of(0, 1)).getContent();
        if (records.isEmpty()) {
            return null;
        }
        OCRHistory record = records.get(0);
        Map<String, Object> sample = new HashMap<>();
        sample.put("id", record.getId());
        sample.put("imageFilename", record.getImageFilename());
        sample.put("recognizedText", record.getRecognizedText());
        sample.put("characterCount", record.getCharacterCount());
        sample.put("confidence", record.getConfidence());
        sample.put("timestamp", record.getTimestamp());
        sample.put("userId", record.getUserId());
        sample.put("isRegistered", record.getIsRegistered());
        return sample;
    }
    
    /**
     * Helper method to get character count from OCRHistory
     * Uses characterCount field if available, otherwise calculates from recognizedText
     */
    private int getCharacterCount(OCRHistory history) {
        if (history.getCharacterCount() != null && history.getCharacterCount() > 0) {
            return history.getCharacterCount();
        }
        // Fallback: calculate from recognizedText
        if (history.getRecognizedText() != null) {
            return history.getRecognizedText().length();
        }
        return 0;
    }
}
