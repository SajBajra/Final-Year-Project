package com.lipika.service.impl;

import com.lipika.model.OCRResponse;
import com.lipika.service.OCRService;
import com.lipika.service.AdminService;
import com.lipika.service.TrialTrackingService;
import com.lipika.util.JwtUtil;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.http.*;
import org.springframework.http.converter.support.AllEncompassingFormHttpMessageConverter;
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
    private final TrialTrackingService trialTrackingService;
    private final JwtUtil jwtUtil;
    
    @Value("${lipika.ocr.service.url:http://localhost:5000}")
    private String ocrServiceUrl;
    
    @Override
    public OCRResponse recognizeText(MultipartFile image, HttpServletRequest request, Long userId) {
        // Extract tracking info
        String ipAddress = getClientIpAddress(request);
        String cookieId = getCookieId(request);
        String fingerprint = generateFingerprint(request);
        
        // Check if user is authenticated
        boolean isAuthenticated = userId != null;
        
        // If not authenticated, check trial limits
        if (!isAuthenticated) {
            if (!trialTrackingService.canPerformOCR(ipAddress, cookieId, fingerprint)) {
                OCRResponse errorResponse = new OCRResponse();
                errorResponse.setSuccess(false);
                errorResponse.setMessage("Trial limit exceeded. Please create an account or login to continue.");
                errorResponse.setTrialInfo(new com.lipika.dto.TrialInfo(
                    0,
                    10,
                    10,
                    true
                ));
                return errorResponse;
            }
        }
        try {
            log.info("Processing OCR request for image: {}", image.getOriginalFilename());
            
            // Prepare multipart request
            // Don't set Content-Type manually - RestTemplate will set it correctly for multipart
            HttpHeaders headers = new HttpHeaders();
            
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            try {
                // Read image bytes
                byte[] imageBytes = image.getBytes();
                
                // Create ByteArrayResource with proper filename
                // This is a Resource that RestTemplate can handle for multipart
                Resource imageResource = new ByteArrayResource(imageBytes) {
                    @Override
                    public String getFilename() {
                        String filename = image.getOriginalFilename();
                        return filename != null && !filename.isEmpty() ? filename : "image.png";
                    }
                };
                
                // Add as HttpEntity with proper ContentType
                String contentType = image.getContentType();
                if (contentType == null || contentType.isEmpty()) {
                    // Guess content type from filename
                    String filename = image.getOriginalFilename();
                    if (filename != null) {
                        if (filename.toLowerCase().endsWith(".png")) {
                            contentType = "image/png";
                        } else if (filename.toLowerCase().endsWith(".jpg") || filename.toLowerCase().endsWith(".jpeg")) {
                            contentType = "image/jpeg";
                        } else if (filename.toLowerCase().endsWith(".webp")) {
                            contentType = "image/webp";
                        } else {
                            contentType = "image/png"; // Default
                        }
                    } else {
                        contentType = "image/png";
                    }
                }
                
                HttpHeaders fileHeaders = new HttpHeaders();
                fileHeaders.setContentType(MediaType.parseMediaType(contentType));
                HttpEntity<Resource> fileEntity = new HttpEntity<>(imageResource, fileHeaders);
                
                body.add("image", fileEntity);
                
                log.debug("Prepared multipart request: filename={}, contentType={}, size={} bytes", 
                    imageResource.getFilename(), contentType, imageBytes.length);
                
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
                
                // Ensure success is set if we have text (Python service might not return success field)
                if (ocrResponse.getText() != null && !ocrResponse.getText().isEmpty()) {
                    ocrResponse.setSuccess(true);
                }
                
                // Log the mapped text to verify UTF-8 handling
                if (ocrResponse.getText() != null) {
                    log.info("Mapped OCR response text: {}", ocrResponse.getText());
                    log.info("Mapped OCR response text length: {}", ocrResponse.getText().length());
                    log.info("OCR Response success: {}, count: {}, confidence: {}", 
                        ocrResponse.isSuccess(), ocrResponse.getCount(), ocrResponse.getConfidence());
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
                
                // Increment trial count if not authenticated
                if (!isAuthenticated) {
                    trialTrackingService.incrementTrialCount(ipAddress, cookieId, fingerprint);
                    int remaining = trialTrackingService.getRemainingTrials(ipAddress, cookieId, fingerprint);
                    ocrResponse.setTrialInfo(new com.lipika.dto.TrialInfo(
                        remaining,
                        10 - remaining,
                        10,
                        remaining == 0
                    ));
                } else {
                    ocrResponse.setTrialInfo(new com.lipika.dto.TrialInfo(
                        null, null, null, false
                    ));
                }
                
                // Save to history if successful
                boolean shouldSave = ocrResponse.isSuccess() && 
                                   ocrResponse.getText() != null && 
                                   !ocrResponse.getText().isEmpty();
                
                log.info("Should save OCR history: success={}, hasText={}, textLength={}", 
                    ocrResponse.isSuccess(), 
                    ocrResponse.getText() != null, 
                    ocrResponse.getText() != null ? ocrResponse.getText().length() : 0);
                
                if (shouldSave) {
                    try {
                        log.info("Saving OCR history: filename={}, userId={}, isRegistered={}", 
                            image.getOriginalFilename(), userId, isAuthenticated);
                        adminService.saveOCRHistory(
                            image.getOriginalFilename(),
                            ocrResponse.getText(),
                            ocrResponse.getCount(),
                            ocrResponse.getConfidence(),
                            userId,
                            isAuthenticated,
                            ipAddress,
                            cookieId
                        );
                        log.info("OCR history saved successfully");
                    } catch (Exception e) {
                        log.error("Failed to save OCR history", e);
                        // Don't fail the request if history save fails
                    }
                } else {
                    log.warn("Skipping OCR history save: success={}, text={}", 
                        ocrResponse.isSuccess(), 
                        ocrResponse.getText() != null ? "present" : "null");
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
    
    private String getClientIpAddress(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("X-Real-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("Proxy-Client-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("WL-Proxy-Client-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getRemoteAddr();
        }
        // Handle multiple IPs (X-Forwarded-For can contain multiple IPs)
        if (ip != null && ip.contains(",")) {
            ip = ip.split(",")[0].trim();
        }
        return ip;
    }
    
    private String getCookieId(HttpServletRequest request) {
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if ("lipika_trial_id".equals(cookie.getName())) {
                    return cookie.getValue();
                }
            }
        }
        // Generate new cookie ID if not found
        return trialTrackingService.generateCookieId();
    }
    
    private String generateFingerprint(HttpServletRequest request) {
        String userAgent = request.getHeader("User-Agent");
        String acceptLanguage = request.getHeader("Accept-Language");
        String acceptEncoding = request.getHeader("Accept-Encoding");
        return trialTrackingService.generateFingerprint(userAgent, acceptLanguage, acceptEncoding);
    }
}
