package com.lipika.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TrialInfo {
    private Integer remainingTrials;
    private Integer usedTrials;
    private Integer maxTrials;
    private Boolean requiresLogin;
}

