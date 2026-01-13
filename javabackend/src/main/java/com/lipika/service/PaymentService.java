package com.lipika.service;

import com.lipika.dto.PaymentInitiateResponse;
import com.lipika.dto.PaymentRequest;
import com.lipika.dto.PaymentVerificationResponse;
import com.lipika.model.Payment;
import com.lipika.model.User;

import java.util.List;

public interface PaymentService {
    PaymentInitiateResponse initiatePayment(PaymentRequest request, User user);
    PaymentVerificationResponse verifyPayment(String data);
    Payment getPaymentByTransactionUuid(String transactionUuid);
    List<Payment> getUserPayments(User user);
    void updatePaymentStatus(String transactionUuid, Payment.PaymentStatus status, String refId);
}
