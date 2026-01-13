package com.lipika.service.impl;

import com.lipika.service.EmailService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailServiceImpl implements EmailService {
    
    private static final Logger logger = LoggerFactory.getLogger(EmailServiceImpl.class);
    
    @Autowired
    private JavaMailSender mailSender;
    
    @Value("${spring.mail.username}")
    private String fromEmail;
    
    @Override
    public void sendOTPEmail(String toEmail, String otp) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(toEmail);
            message.setSubject("Lipika OCR - Password Reset OTP");
            message.setText(
                "Hello,\n\n" +
                "You have requested to reset your password for Lipika OCR.\n\n" +
                "Your OTP (One-Time Password) is: " + otp + "\n\n" +
                "This OTP will expire in 5 minutes.\n\n" +
                "If you did not request this, please ignore this email.\n\n" +
                "Best regards,\n" +
                "Lipika OCR Team"
            );
            
            mailSender.send(message);
            logger.info("OTP email sent successfully to: {}", toEmail);
        } catch (Exception e) {
            logger.error("Failed to send OTP email to: {}", toEmail, e);
            throw new RuntimeException("Failed to send email: " + e.getMessage());
        }
    }
    
    @Override
    public void sendPasswordResetConfirmation(String toEmail) {
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(toEmail);
            message.setSubject("Lipika OCR - Password Reset Successful");
            message.setText(
                "Hello,\n\n" +
                "Your password has been successfully reset.\n\n" +
                "If you did not perform this action, please contact us immediately.\n\n" +
                "Best regards,\n" +
                "Lipika OCR Team"
            );
            
            mailSender.send(message);
            logger.info("Password reset confirmation email sent to: {}", toEmail);
        } catch (Exception e) {
            logger.error("Failed to send password reset confirmation to: {}", toEmail, e);
            // Don't throw exception for confirmation email failure
        }
    }
}
