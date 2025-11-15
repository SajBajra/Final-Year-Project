import axios from 'axios'
import { API_CONFIG, OCR_CONFIG, TRANSLATION_CONFIG } from '../config/constants'

const API_BASE_URL = API_CONFIG.BASE_URL
const OCR_SERVICE_URL = API_CONFIG.OCR_SERVICE_URL

export const recognizeText = async (imageFile, authHeaders = {}, cookieId = null) => {
  try {
    const formData = new FormData()
    formData.append('image', imageFile)
    
    // Prepare headers with auth and cookie
    const headers = {
      'Content-Type': 'multipart/form-data',
      ...authHeaders
    }
    
    // Add cookie to request if provided
    const config = {
      headers,
      timeout: API_CONFIG.OCR_TIMEOUT,
      withCredentials: true // Include cookies in request
    }
    
    // Call Java backend which proxies to Python OCR service
    const response = await axios.post(
      `${API_BASE_URL}/ocr/recognize`,
      formData,
      config
    )
    
    // Java backend returns: { success: true, message: "...", data: { ... } }
    // Extract the data field which contains the OCR response
    if (response.data.success && response.data.data) {
      const ocrData = response.data.data
      
      // Transform Java backend response to match frontend expectations
      return {
        text: ocrData.text || '',
        characters: (ocrData.characters || []).map(char => ({
          character: char.character,
          confidence: char.confidence || 0,
          bbox: char.bbox ? {
            x: char.bbox.x,
            y: char.bbox.y,
            width: char.bbox.width,
            height: char.bbox.height
          } : null,
          index: char.index
        })),
        confidence: ocrData.confidence ? (ocrData.confidence * 100) : 0,
        count: ocrData.count || 0,
        trialInfo: ocrData.trialInfo || null // Include trial info
      }
    } else {
      // Check if it's a trial limit error
      if (response.data.data?.trialInfo?.requiresLogin) {
        return {
          text: '',
          characters: [],
          confidence: 0,
          count: 0,
          trialInfo: response.data.data.trialInfo,
          error: response.data.message || 'Trial limit exceeded'
        }
      }
      throw new Error(response.data.message || 'OCR recognition failed')
    }
    
  } catch (error) {
    console.error('OCR API Error:', error)
    
    if (error.response) {
      throw new Error(`OCR Service Error: ${error.response.data.message || error.response.statusText}`)
    } else if (error.request) {
      throw new Error('OCR Service is not responding. Please check if the Java backend is running on port 8080.')
    } else {
      throw new Error(`Error: ${error.message}`)
    }
  }
}

export const translateText = async (text, targetLanguage = TRANSLATION_CONFIG.DEFAULT_TARGET) => {
  try {
    // Call Java backend translation service
    const response = await axios.post(
      `${API_BASE_URL}/translate`,
      { text, targetLanguage },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: API_CONFIG.TRANSLATION_TIMEOUT
      }
    )
    
    // Java backend returns: { success: true, data: { translatedText: "...", ... } }
    if (response.data.success && response.data.data) {
      const translatedText = response.data.data.translatedText || response.data.data.text
      return translatedText || text
    } else {
      throw new Error(response.data.message || 'Translation failed')
    }
  } catch (error) {
    console.error('Translation service error:', error.message)
    
    // If it's a network error or API error, return original text
    if (error.response) {
      throw new Error(`Translation failed: ${error.response.data?.message || error.response.statusText}`)
    } else if (error.request) {
      throw new Error('Translation service is not responding. Please check if the backend is running.')
    } else {
      throw new Error(`Translation error: ${error.message}`)
    }
  }
}

export const checkHealth = async () => {
  try {
    // Check Java backend health
    const response = await axios.get(`${API_BASE_URL}/health`)
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    return { status: 'error', message: 'Backend service unavailable' }
  }
}


