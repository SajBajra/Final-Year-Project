package com.lipika.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailService {
    
    private static final Logger log = LoggerFactory.getLogger(EmailService.class);
    
    private final JavaMailSender mailSender;
    
    @Value("${spring.mail.username}")
    private String fromEmail;
    
    @Value("${app.frontend.url:http://localhost:5173}")
    private String frontendUrl;
    
    public EmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }
    
    public void sendPasswordResetEmail(String toEmail, String username, String resetToken) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(toEmail);
            message.setSubject("Lipika OCR - Password Reset Request");
            
            String resetLink = frontendUrl + "/reset-password?token=" + resetToken;
            
            String emailBody = "Hello " + username + ",\n\n" +
                    "You requested to reset your password for your Lipika OCR account.\n\n" +
                    "Click the link below to reset your password:\n" +
                    resetLink + "\n\n" +
                    "This link will expire in 1 hour.\n\n" +
                    "If you didn't request this, please ignore this email.\n\n" +
                    "Best regards,\n" +
                    "Lipika OCR Team";
            
            message.setText(emailBody);
            
            mailSender.send(message);
            log.info("Password reset email sent to: {}", toEmail);
        } catch (Exception e) {
            log.error("Failed to send email to: {}", toEmail, e);
            throw new RuntimeException("Failed to send password reset email");
        }
    }
    
    public void sendWelcomeEmail(String toEmail, String username) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(toEmail);
            message.setSubject("Welcome to Lipika OCR!");
            
            String emailBody = "Hello " + username + ",\n\n" +
                    "Welcome to Lipika OCR - Advanced Ranjana Script Recognition!\n\n" +
                    "You have successfully registered your account.\n" +
                    "You can now enjoy 10 free OCR scans. After that, you'll need to upgrade to premium.\n\n" +
                    "Start scanning at: " + frontendUrl + "\n\n" +
                    "Best regards,\n" +
                    "Lipika OCR Team";
            
            message.setText(emailBody);
            
            mailSender.send(message);
            log.info("Welcome email sent to: {}", toEmail);
        } catch (Exception e) {
            log.error("Failed to send welcome email to: {}", toEmail, e);
            // Don't throw exception for welcome email failure
        }
    }
}

