import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api';

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
