package com.lipika.service;

import com.lipika.model.OCRResponse;
import org.springframework.web.multipart.MultipartFile;

import jakarta.servlet.http.HttpServletRequest;

public interface OCRService {
    OCRResponse recognizeText(MultipartFile image, HttpServletRequest request, Long userId);
}
