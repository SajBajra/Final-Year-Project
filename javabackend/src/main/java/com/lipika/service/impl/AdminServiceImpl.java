package com.lipika.service.impl;

import com.lipika.dto.OCRHistoryDTO;
import com.lipika.model.OCRHistory;
import com.lipika.model.Payment;
import com.lipika.model.User;
import com.lipika.repository.OCRHistoryRepository;
import com.lipika.repository.PaymentRepository;
import com.lipika.repository.UserRepository;
import com.lipika.service.AdminService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

@Service
public class AdminServiceImpl implements AdminService {
    
    private static final Logger log = LoggerFactory.getLogger(AdminServiceImpl.class);
    
    private final OCRHistoryRepository ocrHistoryRepository;
    private final PaymentRepository paymentRepository;
    private final UserRepository userRepository;
    
    public AdminServiceImpl(OCRHistoryRepository ocrHistoryRepository, 
                          PaymentRepository paymentRepository,
                          UserRepository userRepository) {
        this.ocrHistoryRepository = ocrHistoryRepository;
        this.paymentRepository = paymentRepository;
        this.userRepository = userRepository;
    }
    
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
    public void saveOCRHistory(String imageFilename, String imagePath, String recognizedText, 
                               Integer characterCount, Double confidence,
                               Long userId, Boolean isRegistered, String ipAddress, String cookieId) {
        OCRHistory history = new OCRHistory();
        history.setImageFilename(imageFilename);
        history.setImagePath(imagePath);
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
        log.info("Saved OCR history to database: ID={}, UserId={}, IsRegistered={}, ImagePath={}, Text={}", 
                saved.getId(), userId, isRegistered, imagePath, recognizedText);
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
        Page<OCRHistoryDTO> historyPage = ocrHistoryRepository.findAllWithUserInfo(pageable);
        
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
        
        Page<OCRHistoryDTO> historyPage;
        
        // Build query based on provided filters (with user info)
        boolean hasSearch = search != null && !search.trim().isEmpty();
        boolean hasDateRange = startDate != null && endDate != null;
        
        if (hasSearch && hasDateRange) {
            // Search + Date range
            historyPage = ocrHistoryRepository.findAllWithUserInfoSearchAndDateRange(
                    search, startDate, endDate, pageable);
        } else if (hasSearch) {
            // Search only
            historyPage = ocrHistoryRepository.findAllWithUserInfoAndSearch(search, pageable);
        } else if (hasDateRange) {
            // Date range only
            historyPage = ocrHistoryRepository.findAllWithUserInfoAndDateRange(
                    startDate, endDate, pageable);
        } else {
            // No filters
            historyPage = ocrHistoryRepository.findAllWithUserInfo(pageable);
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
        List<OCRHistoryDTO> history = (List<OCRHistoryDTO>) filteredResult.get("data");
        
        // Build CSV with user information
        StringBuilder csv = new StringBuilder();
        csv.append("ID,Username,Email,Role,Image Filename,Recognized Text,Character Count,Timestamp,Language\n");
        
        for (OCRHistoryDTO record : history) {
            csv.append(record.getId()).append(",");
            csv.append("\"").append(escapeCSV(record.getUsername() != null ? record.getUsername() : "Guest")).append("\",");
            csv.append("\"").append(escapeCSV(record.getUserEmail() != null ? record.getUserEmail() : "N/A")).append("\",");
            csv.append("\"").append(escapeCSV(record.getUserRole() != null ? record.getUserRole() : "GUEST")).append("\",");
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
    
    /**
     * Get revenue statistics
     */
    @Override
    public Map<String, Object> getRevenueStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        // Total revenue and transactions
        Double totalRevenue = paymentRepository.getTotalRevenue();
        Long completedTransactions = paymentRepository.getCompletedTransactionsCount();
        
        // Monthly revenue (last 30 days)
        LocalDateTime monthAgo = LocalDateTime.now().minusDays(30);
        Double monthlyRevenue = paymentRepository.getRevenueAfterDate(monthAgo);
        Long monthlyTransactions = paymentRepository.getCompletedTransactionsAfterDate(monthAgo);
        
        // Transaction status counts
        Long pendingTransactions = paymentRepository.countByStatus(Payment.PaymentStatus.PENDING);
        Long failedTransactions = paymentRepository.countByStatus(Payment.PaymentStatus.FAILED);
        Long initiatedTransactions = paymentRepository.countByStatus(Payment.PaymentStatus.INITIATED);
        
        stats.put("totalRevenue", totalRevenue != null ? totalRevenue : 0.0);
        stats.put("totalTransactions", completedTransactions != null ? completedTransactions : 0L);
        stats.put("monthlyRevenue", monthlyRevenue != null ? monthlyRevenue : 0.0);
        stats.put("monthlyTransactions", monthlyTransactions != null ? monthlyTransactions : 0L);
        stats.put("pendingTransactions", pendingTransactions != null ? pendingTransactions : 0L);
        stats.put("completedTransactions", completedTransactions != null ? completedTransactions : 0L);
        stats.put("failedTransactions", failedTransactions != null ? failedTransactions : 0L);
        stats.put("initiatedTransactions", initiatedTransactions != null ? initiatedTransactions : 0L);
        
        log.info("Revenue statistics retrieved: Total={}, Monthly={}", totalRevenue, monthlyRevenue);
        
        return stats;
    }
    
    /**
     * Get all users with management details
     */
    @Override
    public List<Map<String, Object>> getAllUsers() {
        List<User> users = userRepository.findAll();
        
        return users.stream().map(user -> {
            Map<String, Object> userMap = new HashMap<>();
            userMap.put("id", user.getId());
            userMap.put("username", user.getUsername());
            userMap.put("email", user.getEmail());
            userMap.put("role", user.getRole());
            userMap.put("isPremium", user.isPremium());
            userMap.put("usageCount", user.getUsageCount());
            userMap.put("usageLimit", user.getUsageLimit());
            userMap.put("createdAt", user.getCreatedAt());
            userMap.put("lastLogin", user.getLastLogin());
            
            // Determine account type
            String accountType;
            if ("ADMIN".equals(user.getRole())) {
                accountType = "Admin";
            } else if (user.isPremium() || "PREMIUM".equals(user.getRole())) {
                accountType = "Paid";
            } else {
                accountType = "Free";
            }
            userMap.put("accountType", accountType);
            
            return userMap;
        }).collect(Collectors.toList());
    }
    
    /**
     * Update user role
     */
    @Override
    @Transactional
    public boolean updateUserRole(Long userId, String role) {
        Optional<User> userOptional = userRepository.findById(userId);
        if (userOptional.isPresent()) {
            User user = userOptional.get();
            user.setRole(role);
            
            // Update premium status based on role
            if ("PREMIUM".equals(role)) {
                user.setPremium(true);
            } else if ("USER".equals(role)) {
                user.setPremium(false);
            }
            // ADMIN role doesn't affect premium status but has unlimited access
            
            userRepository.save(user);
            log.info("Updated user role: userId={}, newRole={}", userId, role);
            return true;
        }
        log.warn("User not found for role update: userId={}", userId);
        return false;
    }
    
    /**
     * Delete user by ID
     */
    @Override
    @Transactional
    public boolean deleteUser(Long userId) {
        if (userRepository.existsById(userId)) {
            userRepository.deleteById(userId);
            log.info("Deleted user: userId={}", userId);
            return true;
        }
        log.warn("User not found for deletion: userId={}", userId);
        return false;
    }
}
