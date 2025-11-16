package com.lipika.repository;

import com.lipika.model.TrialTracking;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface TrialTrackingRepository extends JpaRepository<TrialTracking, Long> {
    @Query("SELECT t FROM TrialTracking t WHERE " +
           "t.ipAddress = :ipAddress AND " +
           "(t.cookieId = :cookieId OR t.fingerprint = :fingerprint)")
    Optional<TrialTracking> findByIpAndCookieOrFingerprint(
        @Param("ipAddress") String ipAddress,
        @Param("cookieId") String cookieId,
        @Param("fingerprint") String fingerprint
    );
    
    Optional<TrialTracking> findByIpAddress(String ipAddress);
    Optional<TrialTracking> findByCookieId(String cookieId);
    Optional<TrialTracking> findByFingerprint(String fingerprint);
}

