package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.User;
import com.lipika.repository.UserRepository;
import com.lipika.repository.OCRHistoryRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/admin/users")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
public class UserManagementController {
    
    private final UserRepository userRepository;
    private final OCRHistoryRepository ocrHistoryRepository;
    
    @GetMapping
    public ResponseEntity<ApiResponse<Map<String, Object>>> getUsers(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            @RequestParam(required = false) String role,
            @RequestParam(required = false) Boolean isActive) {
        
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "createdAt"));
        Page<User> userPage;
        
        if (role != null && isActive != null) {
            userPage = userRepository.findAll(pageable); // You may need to add custom query
        } else {
            userPage = userRepository.findAll(pageable);
        }
        
        Map<String, Object> result = new HashMap<>();
        result.put("data", userPage.getContent());
        result.put("page", page);
        result.put("size", size);
        result.put("total", userPage.getTotalElements());
        result.put("totalPages", userPage.getTotalPages());
        
        return ResponseEntity.ok(ApiResponse.success("Users retrieved successfully", result));
    }
    
    @GetMapping("/stats")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getUserStats() {
        long totalUsers = userRepository.count();
        long registeredUsers = ocrHistoryRepository.countByIsRegistered(true);
        long unregisteredUsers = ocrHistoryRepository.countByIsRegistered(false);
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalUsers", totalUsers);
        stats.put("registeredUsers", registeredUsers);
        stats.put("unregisteredUsers", unregisteredUsers);
        
        return ResponseEntity.ok(ApiResponse.success("User statistics retrieved", stats));
    }
    
    @GetMapping("/{userId}/history")
    public ResponseEntity<ApiResponse<Map<String, Object>>> getUserHistory(
            @PathVariable Long userId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "timestamp"));
        var historyPage = ocrHistoryRepository.findByUserId(userId, pageable);
        
        Map<String, Object> result = new HashMap<>();
        result.put("data", historyPage.getContent());
        result.put("page", page);
        result.put("size", size);
        result.put("total", historyPage.getTotalElements());
        result.put("totalPages", historyPage.getTotalPages());
        
        return ResponseEntity.ok(ApiResponse.success("User history retrieved", result));
    }
    
    @PutMapping("/{userId}/status")
    public ResponseEntity<ApiResponse<User>> updateUserStatus(
            @PathVariable Long userId,
            @RequestParam Boolean isActive) {
        
        return userRepository.findById(userId)
                .map(user -> {
                    user.setIsActive(isActive);
                    user = userRepository.save(user);
                    return ResponseEntity.ok(ApiResponse.success("User status updated", user));
                })
                .orElse(ResponseEntity.notFound().build());
    }
}

