package com.lipika.service.impl;

import com.lipika.model.TranslationRequest;
import com.lipika.model.TranslationResponse;
import com.lipika.service.TranslationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
public class TranslationServiceImpl implements TranslationService {
    
    // Basic Ranjana character to English transliteration mapping
    // This is a simplified mapping - can be enhanced with proper translation service
    private static final Map<String, String> RANJANA_TO_ENGLISH = new HashMap<>();
    
    static {
        // Vowels
        RANJANA_TO_ENGLISH.put("अ", "a");
        RANJANA_TO_ENGLISH.put("आ", "aa");
        RANJANA_TO_ENGLISH.put("इ", "i");
        RANJANA_TO_ENGLISH.put("ई", "ee");
        RANJANA_TO_ENGLISH.put("उ", "u");
        RANJANA_TO_ENGLISH.put("ऊ", "oo");
        RANJANA_TO_ENGLISH.put("ए", "e");
        RANJANA_TO_ENGLISH.put("ऐ", "ai");
        RANJANA_TO_ENGLISH.put("ओ", "o");
        RANJANA_TO_ENGLISH.put("औ", "au");
        
        // Consonants
        RANJANA_TO_ENGLISH.put("क", "ka");
        RANJANA_TO_ENGLISH.put("ख", "kha");
        RANJANA_TO_ENGLISH.put("ग", "ga");
        RANJANA_TO_ENGLISH.put("घ", "gha");
        RANJANA_TO_ENGLISH.put("ङ", "nga");
        RANJANA_TO_ENGLISH.put("च", "cha");
        RANJANA_TO_ENGLISH.put("छ", "chha");
        RANJANA_TO_ENGLISH.put("ज", "ja");
        RANJANA_TO_ENGLISH.put("झ", "jha");
        RANJANA_TO_ENGLISH.put("ञ", "nya");
        RANJANA_TO_ENGLISH.put("ट", "ta");
        RANJANA_TO_ENGLISH.put("ठ", "tha");
        RANJANA_TO_ENGLISH.put("ड", "da");
        RANJANA_TO_ENGLISH.put("ढ", "dha");
        RANJANA_TO_ENGLISH.put("ण", "na");
        RANJANA_TO_ENGLISH.put("त", "ta");
        RANJANA_TO_ENGLISH.put("थ", "tha");
        RANJANA_TO_ENGLISH.put("द", "da");
        RANJANA_TO_ENGLISH.put("ध", "dha");
        RANJANA_TO_ENGLISH.put("न", "na");
        RANJANA_TO_ENGLISH.put("प", "pa");
        RANJANA_TO_ENGLISH.put("फ", "pha");
        RANJANA_TO_ENGLISH.put("ब", "ba");
        RANJANA_TO_ENGLISH.put("भ", "bha");
        RANJANA_TO_ENGLISH.put("म", "ma");
        RANJANA_TO_ENGLISH.put("य", "ya");
        RANJANA_TO_ENGLISH.put("र", "ra");
        RANJANA_TO_ENGLISH.put("ल", "la");
        RANJANA_TO_ENGLISH.put("व", "wa");
        RANJANA_TO_ENGLISH.put("श", "sha");
        RANJANA_TO_ENGLISH.put("ष", "sha");
        RANJANA_TO_ENGLISH.put("स", "sa");
        RANJANA_TO_ENGLISH.put("ह", "ha");
        RANJANA_TO_ENGLISH.put("क्ष", "ksha");
        RANJANA_TO_ENGLISH.put("त्र", "tra");
        RANJANA_TO_ENGLISH.put("ज्ञ", "gya");
        
        // Diacritics
        RANJANA_TO_ENGLISH.put("ा", "aa");
        RANJANA_TO_ENGLISH.put("ि", "i");
        RANJANA_TO_ENGLISH.put("ी", "ee");
        RANJANA_TO_ENGLISH.put("ु", "u");
        RANJANA_TO_ENGLISH.put("ू", "oo");
        RANJANA_TO_ENGLISH.put("े", "e");
        RANJANA_TO_ENGLISH.put("ै", "ai");
        RANJANA_TO_ENGLISH.put("ो", "o");
        RANJANA_TO_ENGLISH.put("ौ", "au");
        RANJANA_TO_ENGLISH.put("्", "");
        RANJANA_TO_ENGLISH.put("ं", "m");
        RANJANA_TO_ENGLISH.put("ः", "h");
        RANJANA_TO_ENGLISH.put("ँ", "n");
        
        // Common words/phrases
        RANJANA_TO_ENGLISH.put("नेपाली", "Nepali");
        RANJANA_TO_ENGLISH.put("भाषा", "language");
        RANJANA_TO_ENGLISH.put("लिपिका", "Lipika");
    }
    
    @Override
    public TranslationResponse translate(TranslationRequest request) {
        try {
            log.info("Translating text: {} to language: {}", request.getText(), request.getTargetLanguage());
            
            String originalText = request.getText();
            String translatedText;
            
            if ("en".equalsIgnoreCase(request.getTargetLanguage())) {
                // Translate to English using transliteration
                translatedText = transliterateToEnglish(originalText);
            } else {
                // For other languages, use transliteration as fallback
                // In production, integrate with Google Translate API or similar
                translatedText = transliterateToEnglish(originalText);
                log.warn("Translation to {} not fully supported, using transliteration", request.getTargetLanguage());
            }
            
            TranslationResponse response = new TranslationResponse();
            response.setOriginalText(originalText);
            response.setTranslatedText(translatedText);
            response.setSourceLanguage("ranjana");
            response.setTargetLanguage(request.getTargetLanguage());
            response.setSuccess(true);
            response.setMessage("Translation successful");
            
            return response;
            
        } catch (Exception e) {
            log.error("Error translating text", e);
            TranslationResponse response = new TranslationResponse();
            response.setSuccess(false);
            response.setMessage("Translation failed: " + e.getMessage());
            return response;
        }
    }
    
    private String transliterateToEnglish(String ranjanaText) {
        if (ranjanaText == null || ranjanaText.isEmpty()) {
            return "";
        }
        
        StringBuilder result = new StringBuilder();
        
        // First, try whole words
        String remainingText = ranjanaText;
        
        // Try to match common words first
        for (Map.Entry<String, String> entry : RANJANA_TO_ENGLISH.entrySet()) {
            if (remainingText.contains(entry.getKey())) {
                // For now, do character-by-character translation
                break;
            }
        }
        
        // Character-by-character transliteration
        for (char c : ranjanaText.toCharArray()) {
            String charStr = String.valueOf(c);
            if (RANJANA_TO_ENGLISH.containsKey(charStr)) {
                result.append(RANJANA_TO_ENGLISH.get(charStr));
            } else if (!Character.isWhitespace(c)) {
                // Unknown character, keep as is
                result.append(charStr);
            } else {
                result.append(" ");
            }
        }
        
        return result.toString().trim();
    }
}
