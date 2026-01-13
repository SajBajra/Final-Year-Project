import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api';

/**
 * Initiate a payment transaction
 */
export const initiatePayment = async (paymentData, authToken) => {
  try {
    const response = await axios.post(
      `${API_URL}/payment/initiate`,
      paymentData,
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Payment initiation error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Verify a payment transaction
 */
export const verifyPayment = async (data) => {
  try {
    const response = await axios.get(
      `${API_URL}/payment/verify`,
      {
        params: { data }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Payment verification error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Get user's payment history
 */
export const getPaymentHistory = async (authToken) => {
  try {
    const response = await axios.get(
      `${API_URL}/payment/history`,
      {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Payment history error:', error);
    throw error.response?.data || error;
  }
};

/**
 * Get payment details by transaction UUID
 */
export const getPaymentDetails = async (transactionUuid) => {
  try {
    const response = await axios.get(
      `${API_URL}/payment/${transactionUuid}`
    );
    return response.data;
  } catch (error) {
    console.error('Payment details error:', error);
    throw error.response?.data || error;
  }
};
