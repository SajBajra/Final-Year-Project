package com.lipika.controller;

import com.lipika.dto.AuthResponse;
import com.lipika.dto.LoginRequest;
import com.lipika.model.ApiResponse;
import com.lipika.service.AuthService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
// This controller is now dedicated to admin authentication only.
// Frontend can point a "secret" admin login page to these URLs (e.g. /api/admin/auth/login).
@RequestMapping("/api/admin/auth")
public class AuthController {
    
    private static final Logger log = LoggerFactory.getLogger(AuthController.class);
    
    private final AuthService authService;
    
    public AuthController(AuthService authService) {
        this.authService = authService;
    }
    
    @PostMapping("/login")
    public ResponseEntity<ApiResponse<AuthResponse>> login(
            @Valid @RequestBody LoginRequest request) {
        try {
            AuthResponse response = authService.login(request);
            return ResponseEntity.ok(ApiResponse.success("Login successful", response));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(ApiResponse.error(e.getMessage()));
        } catch (Exception e) {
            log.error("Error during login", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(ApiResponse.error("Login failed: " + e.getMessage()));
        }
    }
    
    @GetMapping("/me")
    public ResponseEntity<ApiResponse<String>> getCurrentUser() {
        // This endpoint can be used to verify token validity
        return ResponseEntity.ok(ApiResponse.success("Token is valid"));
    }
}

