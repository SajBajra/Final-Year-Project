package com.lipika.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.converter.StringHttpMessageConverter;
import org.springframework.http.converter.json.MappingJackson2HttpMessageConverter;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.reactive.function.client.WebClient;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

@Configuration
public class ApplicationConfig {
    
    @Value("${ocr.service.url:http://localhost:5000}")
    private String ocrServiceUrl;
    
    @Bean
    public RestTemplate restTemplate() {
        RestTemplate restTemplate = new RestTemplate();
        
        // Configure UTF-8 encoding for all message converters
        // This ensures Devanagari and other Unicode characters are properly handled
        MappingJackson2HttpMessageConverter jsonConverter = new MappingJackson2HttpMessageConverter();
        jsonConverter.setDefaultCharset(StandardCharsets.UTF_8);
        
        // Replace existing converters with UTF-8 configured ones
        restTemplate.setMessageConverters(Arrays.asList(
            new StringHttpMessageConverter(StandardCharsets.UTF_8),
            jsonConverter
        ));
        
        return restTemplate;
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
