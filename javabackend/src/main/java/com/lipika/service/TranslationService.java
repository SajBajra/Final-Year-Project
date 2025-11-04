package com.lipika.service;

import com.lipika.model.TranslationRequest;
import com.lipika.model.TranslationResponse;

public interface TranslationService {
    TranslationResponse translate(TranslationRequest request);
}
