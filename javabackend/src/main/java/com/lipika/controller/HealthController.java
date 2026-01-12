package com.lipika.controller;

import com.lipika.model.ApiResponse;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/health")
public class HealthController {
    
    @GetMapping
    public ResponseEntity<ApiResponse<Map<String, Object>>> health() {
        Map<String, Object> healthData = new HashMap<>();
        healthData.put("status", "UP");
        healthData.put("timestamp", LocalDateTime.now().toString());
        healthData.put("service", "Lipika Backend - Presenter Layer");
        healthData.put("version", "1.0.0");
        
        return ResponseEntity.ok(ApiResponse.success("Service is healthy", healthData));
    }
}
