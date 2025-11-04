package com.lipika.controller;

import com.lipika.model.ApiResponse;
import com.lipika.model.TranslationRequest;
import com.lipika.model.TranslationResponse;
import com.lipika.service.TranslationService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequestMapping("/api/translate")
@RequiredArgsConstructor
public class TranslationController {
    
    private final TranslationService translationService;
    
    @PostMapping
    public ResponseEntity<ApiResponse<TranslationResponse>> translate(
            @Valid @RequestBody TranslationRequest request) {
        
        log.info("Received translation request: {} -> {}", request.getText(), request.getTargetLanguage());
        
        try {
            TranslationResponse response = translationService.translate(request);
            
            if (response.isSuccess()) {
                return ResponseEntity.ok(ApiResponse.success("Translation successful", response));
            } else {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error(response.getMessage()));
            }
            
        } catch (Exception e) {
            log.error("Error processing translation request", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error processing translation: " + e.getMessage()));
        }
    }
    
    @PostMapping("/text")
    public ResponseEntity<ApiResponse<String>> translateText(
            @RequestParam String text,
            @RequestParam(defaultValue = "en") String targetLanguage) {
        
        TranslationRequest request = new TranslationRequest(text, targetLanguage);
        
        try {
            TranslationResponse response = translationService.translate(request);
            
            if (response.isSuccess()) {
                return ResponseEntity.ok(ApiResponse.success(response.getTranslatedText()));
            } else {
                return ResponseEntity.badRequest()
                        .body(ApiResponse.error(response.getMessage()));
            }
            
        } catch (Exception e) {
            log.error("Error processing translation request", e);
            return ResponseEntity.internalServerError()
                    .body(ApiResponse.error("Error processing translation: " + e.getMessage()));
        }
    }
}
