package com.lipika.service;

public interface EmailService {
    void sendOTPEmail(String toEmail, String otp);
    void sendPasswordResetConfirmation(String toEmail);
    void sendWelcomeEmail(String toEmail, String username);
    void sendPasswordResetEmail(String toEmail, String username, String token);
}
