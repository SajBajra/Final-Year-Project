package com.lipika.service;

import com.lipika.model.OCRResponse;
import org.springframework.web.multipart.MultipartFile;

public interface OCRService {
    OCRResponse recognizeText(MultipartFile image);
}
