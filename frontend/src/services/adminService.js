import axios from 'axios'
import { API_CONFIG, ADMIN_CONFIG } from '../config/constants'

const ADMIN_API_URL = ADMIN_CONFIG.ADMIN_API_URL || API_CONFIG.BASE_URL + '/admin'

// Helper function to get auth headers from localStorage
const getAuthHeaders = () => {
  const token = localStorage.getItem('token')
  if (!token) return {}
  return {
    'Authorization': `Bearer ${token}`
  }
}

export const getDashboardStats = async () => {
  try {
    const response = await axios.get(`${ADMIN_API_URL}/dashboard/stats`, {
      headers: getAuthHeaders(),
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data.data
  } catch (error) {
    console.error('Error fetching dashboard stats:', error)
    if (error.response) {
      console.error('Response status:', error.response.status)
      console.error('Response data:', error.response.data)
    }
    throw error
  }
}

export const getOCRHistory = async (page = 0, size = 10, filters = {}) => {
  try {
    const params = { page, size, ...filters }
    const response = await axios.get(`${ADMIN_API_URL}/ocr-history`, {
      params,
      headers: getAuthHeaders(),
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data
  } catch (error) {
    console.error('Error fetching OCR history:', error)
    if (error.response) {
      console.error('Response status:', error.response.status)
      console.error('Response data:', error.response.data)
    }
    throw error
  }
}

export const deleteOCRHistory = async (id) => {
  try {
    const response = await axios.delete(`${ADMIN_API_URL}/ocr-history/${id}`, {
      headers: getAuthHeaders(),
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data
  } catch (error) {
    console.error('Error deleting OCR history:', error)
    throw error
  }
}

export const bulkDeleteOCRHistory = async (ids) => {
  try {
    const response = await axios.delete(`${ADMIN_API_URL}/ocr-history/bulk`, {
      data: ids,
      timeout: API_CONFIG.TIMEOUT,
      headers: { 
        'Content-Type': 'application/json',
        ...getAuthHeaders()
      }
    })
    return response.data
  } catch (error) {
    console.error('Error bulk deleting OCR history:', error)
    throw error
  }
}

export const getSettings = async () => {
  try {
    const response = await axios.get(`${ADMIN_API_URL}/settings`, {
      headers: getAuthHeaders(),
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data.data
  } catch (error) {
    console.error('Error fetching settings:', error)
    throw error
  }
}

export const updateSettings = async (settings) => {
  try {
    const response = await axios.put(`${ADMIN_API_URL}/settings`, settings, {
      timeout: API_CONFIG.TIMEOUT,
      headers: { 
        'Content-Type': 'application/json',
        ...getAuthHeaders()
      }
    })
    return response.data
  } catch (error) {
    console.error('Error updating settings:', error)
    throw error
  }
}

export const getAnalytics = async (period = 'daily', days = 30) => {
  try {
    const response = await axios.get(`${ADMIN_API_URL}/analytics`, {
      params: { period, days },
      headers: getAuthHeaders(),
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data
  } catch (error) {
    console.error('Error fetching analytics:', error)
    throw error
  }
}

export const getCharacterStatistics = async () => {
  try {
    const response = await axios.get(`${ADMIN_API_URL}/characters/stats`, {
      headers: getAuthHeaders(),
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data
  } catch (error) {
    console.error('Error fetching character statistics:', error)
    throw error
  }
}

export const exportOCRHistory = async (filters = {}) => {
  try {
    const params = { ...filters }
    const response = await axios.get(`${ADMIN_API_URL}/ocr-history/export`, {
      params,
      headers: getAuthHeaders(),
      responseType: 'blob',
      timeout: API_CONFIG.TIMEOUT * 2 // Longer timeout for exports
    })
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'ocr_history_export.csv')
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
    
    return { success: true }
  } catch (error) {
    console.error('Error exporting OCR history:', error)
    throw error
  }
}

export const changePassword = async (currentPassword, newPassword) => {
  try {
    const response = await axios.post(`${ADMIN_API_URL}/password/change`, {
      currentPassword,
      newPassword
    }, {
      timeout: API_CONFIG.TIMEOUT,
      headers: { 
        'Content-Type': 'application/json',
        ...getAuthHeaders()
      }
    })
    return response.data
  } catch (error) {
    console.error('Error changing password:', error)
    throw error
  }
}

