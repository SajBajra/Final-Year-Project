import axios from 'axios'

const API_BASE_URL = 'http://localhost:8080/api'
const OCR_SERVICE_URL = 'http://localhost:5000' // Python OCR service (used directly if needed)

export const recognizeText = async (imageFile) => {
  try {
    const formData = new FormData()
    formData.append('image', imageFile)
    
    // Call Java backend which proxies to Python OCR service
    const response = await axios.post(
      `${API_BASE_URL}/ocr/recognize`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 30000 // 30 seconds timeout
      }
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
        count: ocrData.count || 0
      }
    } else {
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

export const translateText = async (text, targetLanguage = 'devanagari') => {
  try {
    // Call Java backend translation service
    const response = await axios.post(
      `${API_BASE_URL}/translate`,
      { text, targetLanguage },
      {
        headers: { 'Content-Type': 'application/json' },
        timeout: 10000
      }
    )
    
    // Java backend returns: { success: true, data: { translatedText: "..." } }
    if (response.data.success && response.data.data) {
      return response.data.data.translatedText || text
    } else {
      throw new Error(response.data.message || 'Translation failed')
    }
  } catch (error) {
    console.warn('Translation service unavailable, using fallback:', error.message)
    
    // Fallback: Simple character mapping to Devanagari
    // This is a basic fallback - actual mapping is done in backend
    return text  // Return original if translation fails
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


