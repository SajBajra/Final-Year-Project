package com.lipika.model;

import com.lipika.dto.TrialInfo;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class OCRResponse {
    private boolean success;
    private String text;
    private List<CharacterInfo> characters;
    private Double confidence;
    private int count;
    private String message;
    private TrialInfo trialInfo;
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CharacterInfo {
        private String character;
        private Double confidence;
        private BoundingBox bbox;
        private Integer index;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BoundingBox {
        private Integer x;
        private Integer y;
        private Integer width;
        private Integer height;
    }
}
