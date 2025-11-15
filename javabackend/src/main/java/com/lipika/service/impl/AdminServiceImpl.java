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
    
    // Admin password (in production, use proper password hashing)
    private String adminPassword = "admin"; // Default password
    
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
    public boolean bulkDeleteOCRHistory(List<Long> ids) {
        int deleted = 0;
        for (Long id : ids) {
            if (ocrHistoryStore.remove(id) != null) {
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
        List<OCRHistory> filtered = new ArrayList<>(ocrHistoryStore.values());
        
        // Apply filters
        if (search != null && !search.trim().isEmpty()) {
            String searchLower = search.toLowerCase();
            filtered = filtered.stream()
                    .filter(h -> h.getRecognizedText() != null && 
                            h.getRecognizedText().toLowerCase().contains(searchLower))
                    .collect(Collectors.toList());
        }
        
        if (minConfidence != null) {
            filtered = filtered.stream()
                    .filter(h -> h.getConfidence() != null && h.getConfidence() >= minConfidence)
                    .collect(Collectors.toList());
        }
        
        if (maxConfidence != null) {
            filtered = filtered.stream()
                    .filter(h -> h.getConfidence() != null && h.getConfidence() <= maxConfidence)
                    .collect(Collectors.toList());
        }
        
        if (startDate != null) {
            filtered = filtered.stream()
                    .filter(h -> h.getTimestamp().isAfter(startDate) || h.getTimestamp().isEqual(startDate))
                    .collect(Collectors.toList());
        }
        
        if (endDate != null) {
            filtered = filtered.stream()
                    .filter(h -> h.getTimestamp().isBefore(endDate) || h.getTimestamp().isEqual(endDate))
                    .collect(Collectors.toList());
        }
        
        // Apply sorting
        if (sortBy != null && !sortBy.isEmpty()) {
            boolean ascending = sortOrder == null || "asc".equalsIgnoreCase(sortOrder);
            switch (sortBy.toLowerCase()) {
                case "timestamp":
                    filtered.sort((a, b) -> ascending 
                            ? a.getTimestamp().compareTo(b.getTimestamp())
                            : b.getTimestamp().compareTo(a.getTimestamp()));
                    break;
                case "confidence":
                    filtered.sort((a, b) -> {
                        double aConf = a.getConfidence() != null ? a.getConfidence() : 0.0;
                        double bConf = b.getConfidence() != null ? b.getConfidence() : 0.0;
                        return ascending ? Double.compare(aConf, bConf) : Double.compare(bConf, aConf);
                    });
                    break;
                case "charactercount":
                    filtered.sort((a, b) -> {
                        int aCount = a.getCharacterCount() != null ? a.getCharacterCount() : 0;
                        int bCount = b.getCharacterCount() != null ? b.getCharacterCount() : 0;
                        return ascending ? Integer.compare(aCount, bCount) : Integer.compare(bCount, aCount);
                    });
                    break;
                default:
                    // Default sort by timestamp descending
                    filtered.sort((a, b) -> b.getTimestamp().compareTo(a.getTimestamp()));
            }
        } else {
            // Default sort by timestamp descending
            filtered.sort((a, b) -> b.getTimestamp().compareTo(a.getTimestamp()));
        }
        
        // Pagination
        int start = page * size;
        int end = Math.min(start + size, filtered.size());
        List<OCRHistory> pageData = start < filtered.size() 
                ? filtered.subList(start, end) 
                : new ArrayList<>();
        
        Map<String, Object> result = new HashMap<>();
        result.put("data", pageData);
        result.put("page", page);
        result.put("size", size);
        result.put("total", filtered.size());
        result.put("totalPages", (int) Math.ceil((double) filtered.size() / size));
        
        return result;
    }
    
    @Override
    public Map<String, Object> getAnalytics(String period, int days) {
        Map<String, Object> analytics = new HashMap<>();
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime startDate = now.minusDays(days);
        
        // Filter records within date range
        List<OCRHistory> recentHistory = ocrHistoryStore.values().stream()
                .filter(h -> h.getTimestamp().isAfter(startDate) || h.getTimestamp().isEqual(startDate))
                .collect(Collectors.toList());
        
        // Time series data (daily, weekly, monthly)
        Map<String, Integer> timeSeries = new LinkedHashMap<>();
        Map<String, Double> confidenceSeries = new LinkedHashMap<>();
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
                confidenceSeries.put(key, dayRecords.stream()
                        .filter(h -> h.getConfidence() != null)
                        .mapToDouble(OCRHistory::getConfidence)
                        .average()
                        .orElse(0.0));
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
                confidenceSeries.put(week, weekRecords.stream()
                        .filter(h -> h.getConfidence() != null)
                        .mapToDouble(OCRHistory::getConfidence)
                        .average()
                        .orElse(0.0));
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
                confidenceSeries.put(month, monthRecords.stream()
                        .filter(h -> h.getConfidence() != null)
                        .mapToDouble(OCRHistory::getConfidence)
                        .average()
                        .orElse(0.0));
                characterSeries.put(month, monthRecords.stream()
                        .mapToInt(h -> h.getCharacterCount() != null ? h.getCharacterCount() : 0)
                        .sum());
            }
        }
        
        // Confidence distribution (buckets)
        Map<String, Long> confidenceDistribution = recentHistory.stream()
                .filter(h -> h.getConfidence() != null)
                .collect(Collectors.groupingBy(h -> {
                    double conf = h.getConfidence();
                    if (conf >= 0.9) return "90-100%";
                    if (conf >= 0.8) return "80-90%";
                    if (conf >= 0.7) return "70-80%";
                    if (conf >= 0.6) return "60-70%";
                    return "Below 60%";
                }, Collectors.counting()));
        
        analytics.put("timeSeries", timeSeries);
        analytics.put("confidenceSeries", confidenceSeries);
        analytics.put("characterSeries", characterSeries);
        analytics.put("confidenceDistribution", confidenceDistribution);
        analytics.put("totalRecords", recentHistory.size());
        analytics.put("period", period);
        analytics.put("days", days);
        
        return analytics;
    }
    
    @Override
    public Map<String, Object> getCharacterStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        // Character frequency analysis
        Map<String, Long> characterFrequency = new HashMap<>();
        Map<String, List<Double>> characterConfidences = new HashMap<>();
        
        for (OCRHistory history : ocrHistoryStore.values()) {
            if (history.getRecognizedText() != null) {
                String text = history.getRecognizedText();
                Double confidence = history.getConfidence();
                
                // Count each character
                for (char c : text.toCharArray()) {
                    String charStr = String.valueOf(c);
                    characterFrequency.put(charStr, characterFrequency.getOrDefault(charStr, 0L) + 1);
                    
                    if (confidence != null) {
                        characterConfidences.putIfAbsent(charStr, new ArrayList<>());
                        characterConfidences.get(charStr).add(confidence);
                    }
                }
            }
        }
        
        // Calculate average confidence per character
        Map<String, Double> characterAvgConfidence = new HashMap<>();
        for (Map.Entry<String, List<Double>> entry : characterConfidences.entrySet()) {
            double avg = entry.getValue().stream()
                    .mapToDouble(Double::doubleValue)
                    .average()
                    .orElse(0.0);
            characterAvgConfidence.put(entry.getKey(), avg);
        }
        
        // Get top 20 most frequent characters
        List<Map<String, Object>> topCharacters = characterFrequency.entrySet().stream()
                .sorted((a, b) -> Long.compare(b.getValue(), a.getValue()))
                .limit(20)
                .map(entry -> {
                    Map<String, Object> charStat = new HashMap<>();
                    charStat.put("character", entry.getKey());
                    charStat.put("frequency", entry.getValue());
                    charStat.put("avgConfidence", characterAvgConfidence.getOrDefault(entry.getKey(), 0.0));
                    return charStat;
                })
                .collect(Collectors.toList());
        
        stats.put("totalUniqueCharacters", characterFrequency.size());
        stats.put("topCharacters", topCharacters);
        stats.put("characterFrequency", characterFrequency);
        stats.put("characterAvgConfidence", characterAvgConfidence);
        
        return stats;
    }
    
    @Override
    public String exportOCRHistoryToCSV(String search, Double minConfidence, Double maxConfidence,
                                        LocalDateTime startDate, LocalDateTime endDate) {
        // Use the filtered method to get filtered data
        Map<String, Object> filteredResult = getOCRHistoryFiltered(
                0, Integer.MAX_VALUE, search, minConfidence, maxConfidence, 
                startDate, endDate, "timestamp", "desc");
        
        @SuppressWarnings("unchecked")
        List<OCRHistory> history = (List<OCRHistory>) filteredResult.get("data");
        
        // Build CSV
        StringBuilder csv = new StringBuilder();
        csv.append("ID,Image Filename,Recognized Text,Character Count,Confidence,Timestamp,Language\n");
        
        for (OCRHistory record : history) {
            csv.append(record.getId()).append(",");
            csv.append("\"").append(escapeCSV(record.getImageFilename())).append("\",");
            csv.append("\"").append(escapeCSV(record.getRecognizedText())).append("\",");
            csv.append(record.getCharacterCount() != null ? record.getCharacterCount() : 0).append(",");
            csv.append(record.getConfidence() != null ? String.format("%.4f", record.getConfidence()) : "0.0000").append(",");
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
}

