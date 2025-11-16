package com.lipika.service;

import com.lipika.model.TrialTracking;
import com.lipika.repository.TrialTrackingRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Optional;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class TrialTrackingService {
    
    private static final int MAX_TRIALS = 10;
    private final TrialTrackingRepository trialTrackingRepository;
    
    /**
     * Get or create trial tracking for unregistered user
     */
    @Transactional
    public TrialTracking getOrCreateTracking(String ipAddress, String cookieId, String fingerprint) {
        // Try to find existing tracking
        Optional<TrialTracking> existing = trialTrackingRepository
            .findByIpAndCookieOrFingerprint(ipAddress, cookieId, fingerprint);
        
        if (existing.isPresent()) {
            return existing.get();
        }
        
        // Create new tracking
        TrialTracking tracking = new TrialTracking();
        tracking.setIpAddress(ipAddress);
        tracking.setCookieId(cookieId);
        tracking.setFingerprint(fingerprint);
        tracking.setTrialCount(0);
        tracking.setIsBlocked(false);
        
        return trialTrackingRepository.save(tracking);
    }
    
    /**
     * Check if user can perform OCR (has remaining trials)
     */
    public boolean canPerformOCR(String ipAddress, String cookieId, String fingerprint) {
        TrialTracking tracking = getOrCreateTracking(ipAddress, cookieId, fingerprint);
        
        if (tracking.getIsBlocked()) {
            return false;
        }
        
        return tracking.getTrialCount() < MAX_TRIALS;
    }
    
    /**
     * Get remaining trials
     */
    public int getRemainingTrials(String ipAddress, String cookieId, String fingerprint) {
        TrialTracking tracking = getOrCreateTracking(ipAddress, cookieId, fingerprint);
        
        if (tracking.getIsBlocked()) {
            return 0;
        }
        
        return Math.max(0, MAX_TRIALS - tracking.getTrialCount());
    }
    
    /**
     * Increment trial count
     */
    @Transactional
    public void incrementTrialCount(String ipAddress, String cookieId, String fingerprint) {
        TrialTracking tracking = getOrCreateTracking(ipAddress, cookieId, fingerprint);
        
        tracking.setTrialCount(tracking.getTrialCount() + 1);
        tracking.setLastAttempt(LocalDateTime.now());
        
        // Block if exceeded max trials
        if (tracking.getTrialCount() >= MAX_TRIALS) {
            tracking.setIsBlocked(true);
            log.warn("Trial limit exceeded for IP: {}, Cookie: {}", ipAddress, cookieId);
        }
        
        trialTrackingRepository.save(tracking);
    }
    
    /**
     * Generate or get cookie ID
     */
    public String generateCookieId() {
        return UUID.randomUUID().toString();
    }
    
    /**
     * Generate fingerprint from user agent and other browser info
     */
    public String generateFingerprint(String userAgent, String acceptLanguage, String acceptEncoding) {
        // Simple fingerprint based on browser characteristics
        // In production, you might want to use a more sophisticated fingerprinting library
        String combined = (userAgent != null ? userAgent : "") +
                         (acceptLanguage != null ? acceptLanguage : "") +
                         (acceptEncoding != null ? acceptEncoding : "");
        return String.valueOf(combined.hashCode());
    }
}

