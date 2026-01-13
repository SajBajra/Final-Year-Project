package com.lipika.service;

public interface EmailService {
    void sendOTPEmail(String toEmail, String otp);
    void sendPasswordResetConfirmation(String toEmail);
}
