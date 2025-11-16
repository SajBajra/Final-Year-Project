package com.lipika.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Table(name = "trial_tracking")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class TrialTracking {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "ip_address", nullable = false, length = 45)
    private String ipAddress;

    @Column(name = "cookie_id", length = 255)
    private String cookieId;

    @Column(length = 255)
    private String fingerprint;

    @Column(name = "trial_count", nullable = false)
    private Integer trialCount = 0;

    @Column(name = "first_attempt", updatable = false)
    private LocalDateTime firstAttempt;

    @Column(name = "last_attempt")
    private LocalDateTime lastAttempt;

    @Column(name = "is_blocked", nullable = false)
    private Boolean isBlocked = false;

    @PrePersist
    protected void onCreate() {
        firstAttempt = LocalDateTime.now();
        lastAttempt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        lastAttempt = LocalDateTime.now();
    }
}

