package com.lipika.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class OCRHistory {
    private Long id;
    private String imageFilename;
    private String recognizedText;
    private Integer characterCount;
    private Double confidence;
    private LocalDateTime timestamp;
    private String language;
}

