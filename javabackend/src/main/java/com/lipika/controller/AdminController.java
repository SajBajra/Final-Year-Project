package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.OCRHistory;
import com.lipika.service.AdminService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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
            @RequestParam(defaultValue = "10") int size) {
        try {
            Map<String, Object> history = adminService.getOCRHistory(page, size);
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
}

