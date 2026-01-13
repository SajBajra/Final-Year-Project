import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api';

/**
 * Get dashboard statistics
 */
export const getDashboardStats = async () => {
  try {
    const token = localStorage.getItem('token');
    const response = await axios.get(
      `${API_URL}/admin/dashboard/stats`,
      {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Dashboard stats error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Get analytics data
 */
export const getAnalytics = async (period = 'daily', days = 7) => {
  try {
    const token = localStorage.getItem('token');
    const response = await axios.get(
      `${API_URL}/admin/analytics`,
      {
        params: { period, days },
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Analytics error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Get revenue statistics
 */
export const getRevenueStatistics = async (authToken) => {
  try {
    const response = await axios.get(
      `${API_URL}/admin/revenue/stats`,
      {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Revenue statistics error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Get all users
 */
export const getAllUsers = async (authToken) => {
  try {
    const response = await axios.get(
      `${API_URL}/admin/users`,
      {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Get users error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Update user role
 */
export const updateUserRole = async (userId, role, authToken) => {
  try {
    const response = await axios.put(
      `${API_URL}/admin/users/${userId}/role`,
      { role },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Update user role error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Delete user
 */
export const deleteUser = async (userId, authToken) => {
  try {
    const response = await axios.delete(
      `${API_URL}/admin/users/${userId}`,
      {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Delete user error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Get OCR history with filters
 */
export const getOCRHistory = async (page = 0, size = 10, filters = {}) => {
  try {
    const token = localStorage.getItem('token');
    const response = await axios.get(
      `${API_URL}/admin/ocr-history`,
      {
        params: {
          page,
          size,
          ...filters
        },
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Get OCR history error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Export OCR history to CSV
 */
export const exportOCRHistory = async (filters = {}) => {
  try {
    const token = localStorage.getItem('token');
    const response = await axios.get(
      `${API_URL}/admin/ocr-history/export`,
      {
        params: filters,
        headers: {
          'Authorization': `Bearer ${token}`
        },
        responseType: 'blob'
      }
    );
    return response.data;
  } catch (error) {
    console.error('Export OCR history error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Get character statistics
 */
export const getCharacterStatistics = async () => {
  try {
    const token = localStorage.getItem('token');
    const response = await axios.get(
      `${API_URL}/admin/character-stats`,
      {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Get character statistics error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Change admin password
 */
export const changePassword = async (currentPassword, newPassword) => {
  try {
    const token = localStorage.getItem('token');
    const response = await axios.post(
      `${API_URL}/admin/change-password`,
      {
        currentPassword,
        newPassword
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Change password error:', error);
    throw error.response?.data || error;
  }
};
