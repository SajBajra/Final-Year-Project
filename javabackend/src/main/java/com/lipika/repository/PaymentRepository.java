package com.lipika.repository;

import com.lipika.model.Payment;
import com.lipika.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PaymentRepository extends JpaRepository<Payment, Long> {
    Optional<Payment> findByTransactionUuid(String transactionUuid);
    List<Payment> findByUserOrderByCreatedAtDesc(User user);
    List<Payment> findByStatusOrderByCreatedAtDesc(Payment.PaymentStatus status);
    List<Payment> findByUserAndStatusOrderByCreatedAtDesc(User user, Payment.PaymentStatus status);
}
