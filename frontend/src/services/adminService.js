import axios from 'axios'
import { API_CONFIG, ADMIN_CONFIG } from '../config/constants'

const ADMIN_API_URL = ADMIN_CONFIG.ADMIN_API_URL

export const getDashboardStats = async () => {
  try {
    const response = await axios.get(`${ADMIN_API_URL}/dashboard/stats`, {
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data.data
  } catch (error) {
    console.error('Error fetching dashboard stats:', error)
    throw error
  }
}

export const getOCRHistory = async (page = 0, size = 10) => {
  try {
    const response = await axios.get(`${ADMIN_API_URL}/ocr-history`, {
      params: { page, size },
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data.data
  } catch (error) {
    console.error('Error fetching OCR history:', error)
    throw error
  }
}

export const deleteOCRHistory = async (id) => {
  try {
    const response = await axios.delete(`${ADMIN_API_URL}/ocr-history/${id}`, {
      timeout: API_CONFIG.TIMEOUT
    })
    return response.data
  } catch (error) {
    console.error('Error deleting OCR history:', error)
    throw error
  }
}

export const getSettings = async () => {
  try {
    const response = await axios.get(`${ADMIN_API_URL}/settings`, {
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
      headers: { 'Content-Type': 'application/json' }
    })
    return response.data
  } catch (error) {
    console.error('Error updating settings:', error)
    throw error
  }
}

