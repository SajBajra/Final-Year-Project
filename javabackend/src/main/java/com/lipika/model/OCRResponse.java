package com.lipika.model;

import com.lipika.dto.TrialInfo;
import java.util.List;

public class OCRResponse {
    private boolean success;
    private String text;
    private List<CharacterInfo> characters;
    private Double confidence;
    private int count;
    private String message;
    private TrialInfo trialInfo;
    
    // Constructors
    public OCRResponse() {
    }
    
    public OCRResponse(boolean success, String text, List<CharacterInfo> characters, Double confidence, int count, String message, TrialInfo trialInfo) {
        this.success = success;
        this.text = text;
        this.characters = characters;
        this.confidence = confidence;
        this.count = count;
        this.message = message;
        this.trialInfo = trialInfo;
    }
    
    // Getters and Setters
    public boolean isSuccess() {
        return success;
    }
    
    public void setSuccess(boolean success) {
        this.success = success;
    }
    
    public String getText() {
        return text;
    }
    
    public void setText(String text) {
        this.text = text;
    }
    
    public List<CharacterInfo> getCharacters() {
        return characters;
    }
    
    public void setCharacters(List<CharacterInfo> characters) {
        this.characters = characters;
    }
    
    public Double getConfidence() {
        return confidence;
    }
    
    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }
    
    public int getCount() {
        return count;
    }
    
    public void setCount(int count) {
        this.count = count;
    }
    
    public String getMessage() {
        return message;
    }
    
    public void setMessage(String message) {
        this.message = message;
    }
    
    public TrialInfo getTrialInfo() {
        return trialInfo;
    }
    
    public void setTrialInfo(TrialInfo trialInfo) {
        this.trialInfo = trialInfo;
    }
    
    public static class CharacterInfo {
        private String character;
        private Double confidence;
        private BoundingBox bbox;
        private Integer index;
        
        // Constructors
        public CharacterInfo() {
        }
        
        public CharacterInfo(String character, Double confidence, BoundingBox bbox, Integer index) {
            this.character = character;
            this.confidence = confidence;
            this.bbox = bbox;
            this.index = index;
        }
        
        // Getters and Setters
        public String getCharacter() {
            return character;
        }
        
        public void setCharacter(String character) {
            this.character = character;
        }
        
        public Double getConfidence() {
            return confidence;
        }
        
        public void setConfidence(Double confidence) {
            this.confidence = confidence;
        }
        
        public BoundingBox getBbox() {
            return bbox;
        }
        
        public void setBbox(BoundingBox bbox) {
            this.bbox = bbox;
        }
        
        public Integer getIndex() {
            return index;
        }
        
        public void setIndex(Integer index) {
            this.index = index;
        }
    }
    
    public static class BoundingBox {
        private Integer x;
        private Integer y;
        private Integer width;
        private Integer height;
        
        // Constructors
        public BoundingBox() {
        }
        
        public BoundingBox(Integer x, Integer y, Integer width, Integer height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }
        
        // Getters and Setters
        public Integer getX() {
            return x;
        }
        
        public void setX(Integer x) {
            this.x = x;
        }
        
        public Integer getY() {
            return y;
        }
        
        public void setY(Integer y) {
            this.y = y;
        }
        
        public Integer getWidth() {
            return width;
        }
        
        public void setWidth(Integer width) {
            this.width = width;
        }
        
        public Integer getHeight() {
            return height;
        }
        
        public void setHeight(Integer height) {
            this.height = height;
        }
    }
}
