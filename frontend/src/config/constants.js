// Global configuration constants for the application
// Update these values to change behavior across the entire application

export const API_CONFIG = {
  // Backend API base URL
  BASE_URL: 'http://localhost:8080/api',
  
  // Python OCR service URL (if needed for direct access)
  OCR_SERVICE_URL: 'http://localhost:5000',
  
  // Request timeouts (in milliseconds)
  TIMEOUT: 30000, // 30 seconds
  OCR_TIMEOUT: 60000, // 60 seconds for OCR
  TRANSLATION_TIMEOUT: 30000, // 30 seconds for translation
}

export const UI_CONFIG = {
  // Primary color (used throughout the app) - matches tailwind.config.js primary-600
  PRIMARY_COLOR: '#0d9488', // Deep Teal
  
  // Secondary color - matches tailwind.config.js secondary-500
  SECONDARY_COLOR: '#f97316', // Warm Orange
  
  // Primary color variations for charts
  PRIMARY_LIGHT: '#14b8a6', // primary-500
  PRIMARY_DARK: '#0f766e', // primary-700
  
  // Secondary color variations for charts
  SECONDARY_LIGHT: '#fb923c', // secondary-400
  SECONDARY_DARK: '#ea580c', // secondary-600
  
  // Font family
  FONT_FAMILY: 'Poppins, system-ui, sans-serif',
  
  // Animation durations (in milliseconds)
  ANIMATION_DURATION: 200,
  TRANSITION_DURATION: 300,
  
  // Breakpoints for responsive design
  BREAKPOINTS: {
    SM: '640px',
    MD: '768px',
    LG: '1024px',
    XL: '1280px',
  },
}

export const OCR_CONFIG = {
  // Confidence threshold for character recognition
  MIN_CONFIDENCE: 0.2,
  
  // Maximum file size for upload (in bytes)
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  
  // Supported image formats
  SUPPORTED_FORMATS: ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'],
  
  // Default language for OCR output
  DEFAULT_LANGUAGE: 'devanagari',
}

export const TRANSLATION_CONFIG = {
  // Default source language
  DEFAULT_SOURCE: 'devanagari',
  
  // Default target language
  DEFAULT_TARGET: 'en',
  
  // Enable/disable translation API
  API_ENABLED: true,
}

export const ADMIN_CONFIG = {
  // Admin panel route (no authentication for now)
  ADMIN_ROUTE: '/admin',
  
  // Admin API base URL
  ADMIN_API_URL: 'http://localhost:8080/api/admin',
  
  // Items per page for admin tables
  ITEMS_PER_PAGE: 10,
  
  // Enable/disable admin features
  ENABLED: true,
}

export const ROUTES = {
  HOME: '/',
  FEATURES: '/features',
  ABOUT: '/about',
  PRICING: '/pricing',
  CONTACT: '/contact',
  ADMIN: '/admin',
  ADMIN_DASHBOARD: '/admin/dashboard',
  ADMIN_OCR_HISTORY: '/admin/ocr-history',
  ADMIN_ANALYTICS: '/admin/analytics',
  ADMIN_CHARACTERS: '/admin/characters',
  ADMIN_SETTINGS: '/admin/settings',
  ADMIN_USER_MANAGEMENT: '/admin/users',
}

// Export all configs as a single object for easy access
export default {
  API_CONFIG,
  UI_CONFIG,
  OCR_CONFIG,
  TRANSLATION_CONFIG,
  ADMIN_CONFIG,
  ROUTES,
}

