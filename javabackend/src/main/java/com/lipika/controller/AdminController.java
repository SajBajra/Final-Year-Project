package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.OCRHistory;
import com.lipika.service.AdminService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
public class AdminController {
    
    private final AdminService adminService;
    
    /**
     * Get dashboard statistics
     * GET /api/admin/dashboard/stats
     */
    @GetMapping("/dashboard/stats")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getDashboardStats() {
        try {
            Map<String, Object> stats = adminService.getDashboardStats();
            return ResponseEntity.ok(ApiResponse.success("Dashboard stats retrieved successfully", stats));
        } catch (Exception e) {
            log.error("Error retrieving dashboard stats", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving dashboard stats: " + e.getMessage()));
        }
    }
    
    /**
     * Get OCR history with pagination
     * GET /api/admin/ocr-history?page=0&size=10
     */
    @GetMapping("/ocr-history")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getOCRHistory(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(required = false) String search,
            @RequestParam(required = false) Double minConfidence,
            @RequestParam(required = false) Double maxConfidence,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate,
            @RequestParam(required = false, defaultValue = "timestamp") String sortBy,
            @RequestParam(required = false, defaultValue = "desc") String sortOrder) {
        try {
            Map<String, Object> history;
            if (search != null || minConfidence != null || maxConfidence != null || 
                startDate != null || endDate != null || 
                (sortBy != null && !sortBy.equals("timestamp")) || 
                (sortOrder != null && !sortOrder.equals("desc"))) {
                // Use filtered endpoint
                history = adminService.getOCRHistoryFiltered(
                        page, size, search, minConfidence, maxConfidence, 
                        startDate, endDate, sortBy, sortOrder);
            } else {
                // Use simple pagination
                history = adminService.getOCRHistory(page, size);
            }
            return ResponseEntity.ok(ApiResponse.success("OCR history retrieved successfully", history));
        } catch (Exception e) {
            log.error("Error retrieving OCR history", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving OCR history: " + e.getMessage()));
        }
    }
    
    /**
     * Get OCR history by ID
     * GET /api/admin/ocr-history/{id}
     */
    @GetMapping("/ocr-history/{id}")
    public ResponseEntity<ApiResponse<OCRHistory>> getOCRHistoryById(@PathVariable Long id) {
        try {
            OCRHistory history = adminService.getOCRHistoryById(id);
            if (history != null) {
                return ResponseEntity.ok(ApiResponse.success("OCR history retrieved successfully", history));
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            log.error("Error retrieving OCR history by ID", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving OCR history: " + e.getMessage()));
        }
    }
    
    /**
     * Delete OCR history by ID
     * DELETE /api/admin/ocr-history/{id}
     */
    @DeleteMapping("/ocr-history/{id}")
    public ResponseEntity<ApiResponse<String>> deleteOCRHistory(@PathVariable Long id) {
        try {
            boolean deleted = adminService.deleteOCRHistory(id);
            if (deleted) {
                return ResponseEntity.ok(ApiResponse.success("OCR history deleted successfully", "Deleted"));
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            log.error("Error deleting OCR history", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error deleting OCR history: " + e.getMessage()));
        }
    }
    
    /**
     * Bulk delete OCR history by IDs
     * DELETE /api/admin/ocr-history/bulk
     */
    @DeleteMapping("/ocr-history/bulk")
    public ResponseEntity<ApiResponse<String>> bulkDeleteOCRHistory(@RequestBody List<Long> ids) {
        try {
            boolean deleted = adminService.bulkDeleteOCRHistory(ids);
            if (deleted) {
                return ResponseEntity.ok(ApiResponse.success("OCR history records deleted successfully", "Deleted"));
            } else {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error("No records were deleted"));
            }
        } catch (Exception e) {
            log.error("Error bulk deleting OCR history", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error bulk deleting OCR history: " + e.getMessage()));
        }
    }
    
    /**
     * Get system settings
     * GET /api/admin/settings
     */
    @GetMapping("/settings")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getSettings() {
        try {
            Map<String, Object> settings = adminService.getSettings();
            return ResponseEntity.ok(ApiResponse.success("Settings retrieved successfully", settings));
        } catch (Exception e) {
            log.error("Error retrieving settings", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving settings: " + e.getMessage()));
        }
    }
    
    /**
     * Update system settings
     * PUT /api/admin/settings
     */
    @PutMapping("/settings")
    public ResponseEntity<ApiResponse<String>> updateSettings(@RequestBody Map<String, Object> settings) {
        try {
            boolean updated = adminService.updateSettings(settings);
            if (updated) {
                return ResponseEntity.ok(ApiResponse.success("Settings updated successfully", "Updated"));
            } else {
                return ResponseEntity.internalServerError()
                        .body(ApiResponse.error("Failed to update settings"));
            }
        } catch (Exception e) {
            log.error("Error updating settings", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error updating settings: " + e.getMessage()));
        }
    }
    
    /**
     * Get analytics data for charts
     * GET /api/admin/analytics?period=daily&days=30
     */
    @GetMapping("/analytics")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getAnalytics(
            @RequestParam(defaultValue = "daily") String period,
            @RequestParam(defaultValue = "30") int days) {
        try {
            Map<String, Object> analytics = adminService.getAnalytics(period, days);
            return ResponseEntity.ok(ApiResponse.success("Analytics data retrieved successfully", analytics));
        } catch (Exception e) {
            log.error("Error retrieving analytics", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving analytics: " + e.getMessage()));
        }
    }
    
    /**
     * Get character statistics
     * GET /api/admin/characters/stats
     */
    @GetMapping("/characters/stats")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getCharacterStatistics() {
        try {
            Map<String, Object> stats = adminService.getCharacterStatistics();
            return ResponseEntity.ok(ApiResponse.success("Character statistics retrieved successfully", stats));
        } catch (Exception e) {
            log.error("Error retrieving character statistics", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving character statistics: " + e.getMessage()));
        }
    }
    
    /**
     * Export OCR history to CSV
     * GET /api/admin/ocr-history/export?search=...&minConfidence=...&maxConfidence=...
     */
    @GetMapping("/ocr-history/export")
    public ResponseEntity<String> exportOCRHistoryToCSV(
            @RequestParam(required = false) String search,
            @RequestParam(required = false) Double minConfidence,
            @RequestParam(required = false) Double maxConfidence,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        try {
            String csv = adminService.exportOCRHistoryToCSV(
                    search, minConfidence, maxConfidence, startDate, endDate);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.TEXT_PLAIN);
            headers.setContentDispositionFormData("attachment", "ocr_history_export.csv");
            
            return new ResponseEntity<>(csv, headers, HttpStatus.OK);
        } catch (Exception e) {
            log.error("Error exporting OCR history", e);
            return ResponseEntity.internalServerError()
                    .body("Error exporting OCR history: " + e.getMessage());
        }
    }
}

