package com.lipika.model;

import jakarta.validation.constraints.NotBlank;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TranslationRequest {
    @NotBlank(message = "Text to translate is required")
    private String text;
    
    private String targetLanguage = "en"; // Default to English
}
