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
    
    @Value("${lipika.ocr.service.url:http://localhost:5000}")
    private String ocrServiceUrl;
    
    @Bean
    public RestTemplate restTemplate() {
        RestTemplate restTemplate = new RestTemplate();
        
        // Get default message converters and configure UTF-8
        List<org.springframework.http.converter.HttpMessageConverter<?>> converters = restTemplate.getMessageConverters();
        
        // Configure UTF-8 encoding for String converter
        for (org.springframework.http.converter.HttpMessageConverter<?> converter : converters) {
            if (converter instanceof StringHttpMessageConverter) {
                ((StringHttpMessageConverter) converter).setDefaultCharset(StandardCharsets.UTF_8);
            }
            if (converter instanceof MappingJackson2HttpMessageConverter) {
                ((MappingJackson2HttpMessageConverter) converter).setDefaultCharset(StandardCharsets.UTF_8);
            }
        }
        
        // Ensure FormHttpMessageConverter is present for multipart/form-data
        // This is usually included by default, but we ensure it's there
        boolean hasFormConverter = converters.stream()
            .anyMatch(c -> c instanceof FormHttpMessageConverter || c instanceof AllEncompassingFormHttpMessageConverter);
        
        if (!hasFormConverter) {
            converters.add(new AllEncompassingFormHttpMessageConverter());
        }
        
        restTemplate.setMessageConverters(converters);
        
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
