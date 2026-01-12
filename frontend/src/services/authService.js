import axios from 'axios'

const API_URL = 'http://localhost:8080/api/users'

export const register = async (username, email, password) => {
  const response = await axios.post(`${API_URL}/register`, {
    username,
    email,
    password
  })
  return response.data
}

export const login = async (usernameOrEmail, password) => {
  const response = await axios.post(`${API_URL}/login`, {
    usernameOrEmail,
    password
  })
  return response.data
}

export const forgotPassword = async (email) => {
  const response = await axios.post(`${API_URL}/forgot-password`, {
    email
  })
  return response.data
}

export const resetPassword = async (token, newPassword) => {
  const response = await axios.post(`${API_URL}/reset-password`, {
    token,
    newPassword
  })
  return response.data
}

export const getUserProfile = async (token) => {
  const response = await axios.get(`${API_URL}/profile`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  })
  return response.data
}

export const checkUsageStatus = async (token) => {
  const response = await axios.get(`${API_URL}/usage-status`, {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  })
  return response.data
}

