package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.OCRHistory;
import com.lipika.service.AdminService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin")
public class AdminController {
    
    private static final Logger log = LoggerFactory.getLogger(AdminController.class);
    
    private final AdminService adminService;
    
    public AdminController(AdminService adminService) {
        this.adminService = adminService;
    }
    
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
     * Diagnostic endpoint to check database connection and record count
     * GET /api/admin/diagnostics
     */
    @GetMapping("/diagnostics")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getDiagnostics() {
        try {
            Map<String, Object> diagnostics = new HashMap<>();
            
            // Check total record count
            long totalRecords = adminService.getTotalRecordCount();
            diagnostics.put("totalRecords", totalRecords);
            
            // Check database connection
            diagnostics.put("databaseConnected", true);
            
            // Get sample record if exists
            if (totalRecords > 0) {
                Map<String, Object> sample = adminService.getSampleRecord();
                diagnostics.put("sampleRecord", sample);
            } else {
                diagnostics.put("sampleRecord", null);
                diagnostics.put("message", "No OCR records found. Perform an OCR operation to create history.");
            }
            
            return ResponseEntity.ok(ApiResponse.success("Diagnostics retrieved", diagnostics));
        } catch (Exception e) {
            log.error("Error getting diagnostics", e);
            Map<String, Object> error = new HashMap<>();
            error.put("databaseConnected", false);
            error.put("error", e.getMessage());
            return ResponseEntity.ok(ApiResponse.error("Error getting diagnostics: " + e.getMessage(), error));
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
            
            // Check if any actual filters are provided (not just defaults)
            boolean hasFilters = (search != null && !search.trim().isEmpty()) || 
                               minConfidence != null || 
                               maxConfidence != null || 
                               startDate != null || 
                               endDate != null;
            
            // Check if sort differs from defaults
            boolean hasCustomSort = (sortBy != null && !sortBy.equals("timestamp")) || 
                                   (sortOrder != null && !sortOrder.equals("desc"));
            
            log.info("OCR History request: page={}, size={}, hasFilters={}, hasCustomSort={}, search={}, sortBy={}, sortOrder={}", 
                page, size, hasFilters, hasCustomSort, search, sortBy, sortOrder);
            
            if (hasFilters || hasCustomSort) {
                // Use filtered endpoint
                history = adminService.getOCRHistoryFiltered(
                        page, size, search, minConfidence, maxConfidence, 
                        startDate, endDate, sortBy, sortOrder);
                log.info("Using filtered query, returned {} records", 
                    history.get("data") != null ? ((java.util.List<?>) history.get("data")).size() : 0);
            } else {
                // Use simple pagination
                history = adminService.getOCRHistory(page, size);
                log.info("Using simple query, returned {} records, total={}", 
                    history.get("data") != null ? ((java.util.List<?>) history.get("data")).size() : 0,
                    history.get("total"));
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
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        try {
            String csv = adminService.exportOCRHistoryToCSV(
                    search, startDate, endDate);
            
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
    
    /**
     * Change admin password
     * POST /api/admin/password/change
     */
    @PostMapping("/password/change")
    public ResponseEntity<ApiResponse<String>> changePassword(@RequestBody Map<String, String> request) {
        try {
            String currentPassword = request.get("currentPassword");
            String newPassword = request.get("newPassword");
            
            if (currentPassword == null || newPassword == null) {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error("Current password and new password are required"));
            }
            
            boolean changed = adminService.changePassword(currentPassword, newPassword);
            if (changed) {
                return ResponseEntity.ok(ApiResponse.success("Password changed successfully", "Password updated"));
            } else {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error("Failed to change password. Please check your current password and ensure new password is at least 4 characters."));
            }
        } catch (Exception e) {
            log.error("Error changing password", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error changing password: " + e.getMessage()));
        }
    }
    
    /**
     * Get revenue statistics
     * GET /api/admin/revenue/stats
     */
    @GetMapping("/revenue/stats")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getRevenueStatistics() {
        try {
            Map<String, Object> stats = adminService.getRevenueStatistics();
            return ResponseEntity.ok(ApiResponse.success("Revenue statistics retrieved successfully", stats));
        } catch (Exception e) {
            log.error("Error retrieving revenue statistics", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving revenue statistics: " + e.getMessage()));
        }
    }
    
    /**
     * Get all users with management details
     * GET /api/admin/users
     */
    @GetMapping("/users")
    public ResponseEntity<ApiResponse<List<Map<String, Object>>>> getAllUsers() {
        try {
            List<Map<String, Object>> users = adminService.getAllUsers();
            return ResponseEntity.ok(ApiResponse.success("Users retrieved successfully", users));
        } catch (Exception e) {
            log.error("Error retrieving users", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving users: " + e.getMessage()));
        }
    }
    
    /**
     * Update user role
     * PUT /api/admin/users/{userId}/role
     */
    @PutMapping("/users/{userId}/role")
    public ResponseEntity<ApiResponse<String>> updateUserRole(
            @PathVariable Long userId,
            @RequestBody Map<String, String> request) {
        try {
            String role = request.get("role");
            if (role == null || role.trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error("Role is required"));
            }
            
            boolean updated = adminService.updateUserRole(userId, role);
            if (updated) {
                return ResponseEntity.ok(ApiResponse.success("User role updated successfully", "Role updated to " + role));
            } else {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error("Failed to update user role. User may not exist."));
            }
        } catch (Exception e) {
            log.error("Error updating user role", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error updating user role: " + e.getMessage()));
        }
    }
    
    /**
     * Delete user
     * DELETE /api/admin/users/{userId}
     */
    @DeleteMapping("/users/{userId}")
    public ResponseEntity<ApiResponse<String>> deleteUser(@PathVariable Long userId) {
        try {
            boolean deleted = adminService.deleteUser(userId);
            if (deleted) {
                return ResponseEntity.ok(ApiResponse.success("User deleted successfully", "User removed"));
            } else {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error("Failed to delete user. User may not exist."));
            }
        } catch (Exception e) {
            log.error("Error deleting user", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error deleting user: " + e.getMessage()));
        }
    }
    
    /**
     * Get all contact form submissions
     * GET /api/admin/contacts
     */
    @GetMapping("/contacts")
    public ResponseEntity<ApiResponse<List<Map<String, Object>>>> getAllContacts() {
        try {
            List<Map<String, Object>> contacts = adminService.getAllContacts();
            return ResponseEntity.ok(ApiResponse.success("Contact submissions retrieved successfully", contacts));
        } catch (Exception e) {
            log.error("Error retrieving contacts", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error retrieving contacts: " + e.getMessage()));
        }
    }
    
    /**
     * Mark contact as read
     * PUT /api/admin/contacts/{contactId}/read
     */
    @PutMapping("/contacts/{contactId}/read")
    public ResponseEntity<ApiResponse<String>> markContactAsRead(@PathVariable Long contactId) {
        try {
            boolean marked = adminService.markContactAsRead(contactId);
            if (marked) {
                return ResponseEntity.ok(ApiResponse.success("Contact marked as read", "Success"));
            } else {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error("Failed to mark contact as read. Contact may not exist."));
            }
        } catch (Exception e) {
            log.error("Error marking contact as read", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error marking contact as read: " + e.getMessage()));
        }
    }
    
    /**
     * Delete contact submission
     * DELETE /api/admin/contacts/{contactId}
     */
    @DeleteMapping("/contacts/{contactId}")
    public ResponseEntity<ApiResponse<String>> deleteContact(@PathVariable Long contactId) {
        try {
            boolean deleted = adminService.deleteContact(contactId);
            if (deleted) {
                return ResponseEntity.ok(ApiResponse.success("Contact deleted successfully", "Success"));
            } else {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error("Failed to delete contact. Contact may not exist."));
            }
        } catch (Exception e) {
            log.error("Error deleting contact", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error deleting contact: " + e.getMessage()));
        }
    }
}

