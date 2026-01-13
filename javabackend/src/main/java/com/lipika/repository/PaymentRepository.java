package com.lipika.repository;

import com.lipika.model.Payment;
import com.lipika.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface PaymentRepository extends JpaRepository<Payment, Long> {
    Optional<Payment> findByTransactionUuid(String transactionUuid);
    List<Payment> findByUserOrderByCreatedAtDesc(User user);
    List<Payment> findByStatusOrderByCreatedAtDesc(Payment.PaymentStatus status);
    List<Payment> findByUserAndStatusOrderByCreatedAtDesc(User user, Payment.PaymentStatus status);
    
    // Revenue queries
    @Query("SELECT COALESCE(SUM(p.totalAmount), 0.0) FROM Payment p WHERE p.status = 'COMPLETED'")
    Double getTotalRevenue();
    
    @Query("SELECT COUNT(p) FROM Payment p WHERE p.status = 'COMPLETED'")
    Long getCompletedTransactionsCount();
    
    @Query("SELECT COALESCE(SUM(p.totalAmount), 0.0) FROM Payment p WHERE p.status = 'COMPLETED' AND p.createdAt >= :startDate")
    Double getRevenueAfterDate(@Param("startDate") LocalDateTime startDate);
    
    @Query("SELECT COUNT(p) FROM Payment p WHERE p.status = 'COMPLETED' AND p.createdAt >= :startDate")
    Long getCompletedTransactionsAfterDate(@Param("startDate") LocalDateTime startDate);
    
    Long countByStatus(Payment.PaymentStatus status);
}
