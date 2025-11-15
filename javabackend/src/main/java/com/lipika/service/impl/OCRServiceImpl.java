package com.lipika.service.impl;

import com.lipika.model.OCRResponse;
import com.lipika.service.OCRService;
import com.lipika.service.AdminService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class OCRServiceImpl implements OCRService {
    
    private final RestTemplate restTemplate;
    private final AdminService adminService;
    
    @Value("${lipika.ocr.service.url:http://localhost:5000}")
    private String ocrServiceUrl;
    
    @Override
    public OCRResponse recognizeText(MultipartFile image) {
        try {
            log.info("Processing OCR request for image: {}", image.getOriginalFilename());
            
            // Prepare multipart request
            HttpHeaders headers = new HttpHeaders();
            // Don't set Content-Type manually - RestTemplate will set it correctly for multipart
            
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            try {
                // Create ByteArrayResource with proper filename
                ByteArrayResource imageResource = new ByteArrayResource(image.getBytes()) {
                    @Override
                    public String getFilename() {
                        return image.getOriginalFilename() != null ? image.getOriginalFilename() : "image.png";
                    }
                };
                body.add("image", imageResource);
            } catch (Exception e) {
                log.error("Error reading image file", e);
                return createErrorResponse("Error reading image file: " + e.getMessage());
            }
            
            // Create HttpEntity - RestTemplate will automatically set multipart/form-data Content-Type
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
            
            // Call Python OCR service
            String url = ocrServiceUrl + "/predict";
            log.info("Calling OCR service at: {}", url);
            
            ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
                url,
                HttpMethod.POST,
                requestEntity,
                new ParameterizedTypeReference<Map<String, Object>>() {}
            );
            
            if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
                Map<String, Object> responseBody = response.getBody();
                
                // Log the raw response text for debugging
                Object rawText = responseBody.get("text");
                if (rawText != null) {
                    log.info("Raw OCR response text (before mapping): {}", rawText);
                    log.info("Raw OCR response text type: {}", rawText.getClass().getName());
                    log.info("Raw OCR response text length: {}", rawText.toString().length());
                }
                
                OCRResponse ocrResponse = mapToOCRResponse(responseBody);
                
                // Log the mapped text to verify UTF-8 handling
                if (ocrResponse.getText() != null) {
                    log.info("Mapped OCR response text: {}", ocrResponse.getText());
                    log.info("Mapped OCR response text length: {}", ocrResponse.getText().length());
                    // Log first few characters as Unicode code points
                    if (!ocrResponse.getText().isEmpty()) {
                        String firstChars = ocrResponse.getText().substring(0, Math.min(10, ocrResponse.getText().length()));
                        StringBuilder unicodeInfo = new StringBuilder("Unicode codes: ");
                        for (char c : firstChars.toCharArray()) {
                            unicodeInfo.append(String.format("U+%04X ", (int) c));
                        }
                        log.info(unicodeInfo.toString());
                    }
                }
                
                // Save to history if successful
                if (ocrResponse.isSuccess() && ocrResponse.getText() != null && !ocrResponse.getText().isEmpty()) {
                    try {
                        adminService.saveOCRHistory(
                            image.getOriginalFilename(),
                            ocrResponse.getText(),
                            ocrResponse.getCount(),
                            ocrResponse.getConfidence()
                        );
                    } catch (Exception e) {
                        log.warn("Failed to save OCR history", e);
                        // Don't fail the request if history save fails
                    }
                }
                
                return ocrResponse;
            } else {
                return createErrorResponse("OCR service returned error status: " + response.getStatusCode());
            }
            
        } catch (RestClientException e) {
            log.error("Error calling OCR service", e);
            return createErrorResponse("OCR service unavailable: " + e.getMessage());
        } catch (Exception e) {
            log.error("Unexpected error in OCR service", e);
            return createErrorResponse("Unexpected error: " + e.getMessage());
        }
    }
    
    @SuppressWarnings("unchecked")
    private OCRResponse mapToOCRResponse(Map<String, Object> responseBody) {
        OCRResponse response = new OCRResponse();
        
        response.setSuccess((Boolean) responseBody.getOrDefault("success", false));
        
        // Properly extract text with UTF-8 encoding support
        Object textObj = responseBody.get("text");
        String text = null;
        if (textObj != null) {
            if (textObj instanceof String) {
                text = (String) textObj;
            } else {
                // Convert to string while preserving UTF-8 characters
                text = textObj.toString();
            }
        }
        response.setText(text);
        
        // Handle count - can be Integer or Long from JSON
        Object countObj = responseBody.get("count");
        if (countObj instanceof Number) {
            response.setCount(((Number) countObj).intValue());
        } else {
            response.setCount(0);
        }
        
        // Handle confidence - can be Double or other Number from JSON
        Object conf = responseBody.get("confidence");
        if (conf instanceof Number) {
            response.setConfidence(((Number) conf).doubleValue());
        }
        
        // Map characters - handle words if present
        if (responseBody.get("characters") instanceof java.util.List) {
            java.util.List<Map<String, Object>> chars = (java.util.List<Map<String, Object>>) responseBody.get("characters");
            response.setCharacters(chars.stream()
                .map(this::mapToCharacterInfo)
                .filter(charInfo -> charInfo.getCharacter() != null)
                .toList());
        }
        
        return response;
    }
    
    @SuppressWarnings("unchecked")
    private OCRResponse.CharacterInfo mapToCharacterInfo(Map<String, Object> charMap) {
        OCRResponse.CharacterInfo charInfo = new OCRResponse.CharacterInfo();
        charInfo.setCharacter((String) charMap.get("character"));
        
        if (charMap.get("confidence") != null) {
            Object conf = charMap.get("confidence");
            if (conf instanceof Number) {
                charInfo.setConfidence(((Number) conf).doubleValue());
            }
        }
        
        if (charMap.get("index") != null) {
            Object idx = charMap.get("index");
            if (idx instanceof Number) {
                charInfo.setIndex(((Number) idx).intValue());
            }
        }
        
        // Map bounding box
        if (charMap.get("bbox") instanceof Map) {
            Map<String, Object> bboxMap = (Map<String, Object>) charMap.get("bbox");
            OCRResponse.BoundingBox bbox = new OCRResponse.BoundingBox();
            
            if (bboxMap.get("x") instanceof Number) {
                bbox.setX(((Number) bboxMap.get("x")).intValue());
            }
            if (bboxMap.get("y") instanceof Number) {
                bbox.setY(((Number) bboxMap.get("y")).intValue());
            }
            if (bboxMap.get("width") instanceof Number) {
                bbox.setWidth(((Number) bboxMap.get("width")).intValue());
            }
            if (bboxMap.get("height") instanceof Number) {
                bbox.setHeight(((Number) bboxMap.get("height")).intValue());
            }
            
            charInfo.setBbox(bbox);
        }
        
        return charInfo;
    }
    
    private OCRResponse createErrorResponse(String message) {
        OCRResponse response = new OCRResponse();
        response.setSuccess(false);
        response.setMessage(message);
        response.setCount(0);
        return response;
    }
}
