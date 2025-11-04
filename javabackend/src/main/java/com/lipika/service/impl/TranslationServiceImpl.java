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
    
    // Ranjana to Devanagari mapping (comprehensive character mapping)
    private static final Map<String, String> RANJANA_TO_DEVANAGARI = new HashMap<>();
    
    // Basic Ranjana character to English transliteration mapping (for fallback)
    private static final Map<String, String> RANJANA_TO_ENGLISH = new HashMap<>();
    
    static {
        // ============================================
        // RANJANA TO DEVANAGARI MAPPING
        // ============================================
        
        // Vowels (स्वर)
        RANJANA_TO_DEVANAGARI.put("अ", "अ");
        RANJANA_TO_DEVANAGARI.put("आ", "आ");
        RANJANA_TO_DEVANAGARI.put("इ", "इ");
        RANJANA_TO_DEVANAGARI.put("ई", "ई");
        RANJANA_TO_DEVANAGARI.put("उ", "उ");
        RANJANA_TO_DEVANAGARI.put("ऊ", "ऊ");
        RANJANA_TO_DEVANAGARI.put("ए", "ए");
        RANJANA_TO_DEVANAGARI.put("ऐ", "ऐ");
        RANJANA_TO_DEVANAGARI.put("ओ", "ओ");
        RANJANA_TO_DEVANAGARI.put("औ", "औ");
        
        // Consonants (व्यंजन)
        RANJANA_TO_DEVANAGARI.put("क", "क");
        RANJANA_TO_DEVANAGARI.put("ख", "ख");
        RANJANA_TO_DEVANAGARI.put("ग", "ग");
        RANJANA_TO_DEVANAGARI.put("घ", "घ");
        RANJANA_TO_DEVANAGARI.put("ङ", "ङ");
        RANJANA_TO_DEVANAGARI.put("च", "च");
        RANJANA_TO_DEVANAGARI.put("छ", "छ");
        RANJANA_TO_DEVANAGARI.put("ज", "ज");
        RANJANA_TO_DEVANAGARI.put("झ", "झ");
        RANJANA_TO_DEVANAGARI.put("ञ", "ञ");
        RANJANA_TO_DEVANAGARI.put("ट", "ट");
        RANJANA_TO_DEVANAGARI.put("ठ", "ठ");
        RANJANA_TO_DEVANAGARI.put("ड", "ड");
        RANJANA_TO_DEVANAGARI.put("ढ", "ढ");
        RANJANA_TO_DEVANAGARI.put("ण", "ण");
        RANJANA_TO_DEVANAGARI.put("त", "त");
        RANJANA_TO_DEVANAGARI.put("थ", "थ");
        RANJANA_TO_DEVANAGARI.put("द", "द");
        RANJANA_TO_DEVANAGARI.put("ध", "ध");
        RANJANA_TO_DEVANAGARI.put("न", "न");
        RANJANA_TO_DEVANAGARI.put("प", "प");
        RANJANA_TO_DEVANAGARI.put("फ", "फ");
        RANJANA_TO_DEVANAGARI.put("ब", "ब");
        RANJANA_TO_DEVANAGARI.put("भ", "भ");
        RANJANA_TO_DEVANAGARI.put("म", "म");
        RANJANA_TO_DEVANAGARI.put("य", "य");
        RANJANA_TO_DEVANAGARI.put("र", "र");
        RANJANA_TO_DEVANAGARI.put("ल", "ल");
        RANJANA_TO_DEVANAGARI.put("व", "व");
        RANJANA_TO_DEVANAGARI.put("श", "श");
        RANJANA_TO_DEVANAGARI.put("ष", "ष");
        RANJANA_TO_DEVANAGARI.put("स", "स");
        RANJANA_TO_DEVANAGARI.put("ह", "ह");
        RANJANA_TO_DEVANAGARI.put("क्ष", "क्ष");
        RANJANA_TO_DEVANAGARI.put("त्र", "त्र");
        RANJANA_TO_DEVANAGARI.put("ज्ञ", "ज्ञ");
        
        // Diacritics/Matra (मात्रा)
        RANJANA_TO_DEVANAGARI.put("ा", "ा");  // aa
        RANJANA_TO_DEVANAGARI.put("ि", "ि");  // i
        RANJANA_TO_DEVANAGARI.put("ी", "ी");  // ii
        RANJANA_TO_DEVANAGARI.put("ु", "ु");  // u
        RANJANA_TO_DEVANAGARI.put("ू", "ू");  // uu
        RANJANA_TO_DEVANAGARI.put("े", "े");  // e
        RANJANA_TO_DEVANAGARI.put("ै", "ै");  // ai
        RANJANA_TO_DEVANAGARI.put("ो", "ो");  // o
        RANJANA_TO_DEVANAGARI.put("ौ", "ौ");  // au
        RANJANA_TO_DEVANAGARI.put("्", "्");  // halant/virām
        RANJANA_TO_DEVANAGARI.put("ं", "ं");  // anusvāra
        RANJANA_TO_DEVANAGARI.put("ः", "ः");  // visarga
        RANJANA_TO_DEVANAGARI.put("ँ", "ँ");  // chandrabindu
        
        // Common compound characters
        RANJANA_TO_DEVANAGARI.put("्र", "्र");  // ra
        RANJANA_TO_DEVANAGARI.put("्य", "्य");  // ya
        RANJANA_TO_DEVANAGARI.put("्व", "्व");  // va
        
        // Numbers (if present)
        RANJANA_TO_DEVANAGARI.put("०", "०");  // 0
        RANJANA_TO_DEVANAGARI.put("१", "१");  // 1
        RANJANA_TO_DEVANAGARI.put("२", "२");  // 2
        RANJANA_TO_DEVANAGARI.put("३", "३");  // 3
        RANJANA_TO_DEVANAGARI.put("४", "४");  // 4
        RANJANA_TO_DEVANAGARI.put("५", "५");  // 5
        RANJANA_TO_DEVANAGARI.put("६", "६");  // 6
        RANJANA_TO_DEVANAGARI.put("७", "७");  // 7
        RANJANA_TO_DEVANAGARI.put("८", "८");  // 8
        RANJANA_TO_DEVANAGARI.put("९", "९");  // 9
        
        // ============================================
        // RANJANA TO ENGLISH MAPPING (for fallback)
        // ============================================
        
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
            String targetLang = request.getTargetLanguage().toLowerCase();
            
            if ("devanagari".equalsIgnoreCase(targetLang) || "dev".equalsIgnoreCase(targetLang) || "hi".equalsIgnoreCase(targetLang)) {
                // Translate Ranjana to Devanagari
                translatedText = transliterateToDevanagari(originalText);
                log.info("Translated {} characters to Devanagari", translatedText.length());
            } else if ("en".equalsIgnoreCase(targetLang)) {
                // Translate to English using transliteration
                translatedText = transliterateToEnglish(originalText);
            } else {
                // Default to Devanagari (as per user requirement)
                translatedText = transliterateToDevanagari(originalText);
                log.warn("Unknown target language '{}', defaulting to Devanagari", request.getTargetLanguage());
            }
            
            TranslationResponse response = new TranslationResponse();
            response.setOriginalText(originalText);
            response.setTranslatedText(translatedText);
            response.setSourceLanguage("ranjana");
            response.setTargetLanguage(targetLang);
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
    
    /**
     * Transliterate Ranjana script to Devanagari script
     * Character-by-character mapping preserving the original text structure
     */
    private String transliterateToDevanagari(String ranjanaText) {
        if (ranjanaText == null || ranjanaText.isEmpty()) {
            return "";
        }
        
        StringBuilder result = new StringBuilder();
        
        // Character-by-character transliteration
        for (char c : ranjanaText.toCharArray()) {
            String charStr = String.valueOf(c);
            
            if (RANJANA_TO_DEVANAGARI.containsKey(charStr)) {
                result.append(RANJANA_TO_DEVANAGARI.get(charStr));
            } else if (Character.isWhitespace(c)) {
                result.append(" ");
            } else {
                // Unknown character, keep as is (might be punctuation, numbers, etc.)
                result.append(charStr);
            }
        }
        
        return result.toString().trim();
    }
    
    /**
     * Transliterate Ranjana script to English (fallback method)
     */
    private String transliterateToEnglish(String ranjanaText) {
        if (ranjanaText == null || ranjanaText.isEmpty()) {
            return "";
        }
        
        StringBuilder result = new StringBuilder();
        
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
