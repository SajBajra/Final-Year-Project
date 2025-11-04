package com.lipika.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class ApplicationConfig {
    
    @Value("${ocr.service.url:http://localhost:5000}")
    private String ocrServiceUrl;
    
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
    
    @Bean
    public WebClient webClient() {
        return WebClient.builder()
                .baseUrl(ocrServiceUrl)
                .build();
    }
    
    @Bean
    public String ocrServiceUrl() {
        return ocrServiceUrl;
    }
}
