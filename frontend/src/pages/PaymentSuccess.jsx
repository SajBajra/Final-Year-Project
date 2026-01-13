import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaCheckCircle, FaSpinner } from 'react-icons/fa';
import axios from 'axios';
import CryptoJS from 'crypto-js';

const PaymentSuccess = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [verifying, setVerifying] = useState(true);
  const [verified, setVerified] = useState(false);
  const [error, setError] = useState(null);
  
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api';
  
  useEffect(() => {
    verifyPayment();
  }, []);
  
  const verifyPayment = async () => {
    try {
      // Get payment data from URL params (returned by eSewa)
      // eSewa returns parameters like: ?data=base64EncodedData or individual params
      const data = searchParams.get('data');
      
      let paymentData;
      
      if (data) {
        // If data is base64 encoded
        try {
          const decodedData = atob(data);
          paymentData = JSON.parse(decodedData);
        } catch (e) {
          console.error('Error decoding data:', e);
          setError('Invalid payment response format');
          setVerifying(false);
          return;
        }
      } else {
        // eSewa might return individual query parameters
        paymentData = {
          status: searchParams.get('status'),
          transaction_code: searchParams.get('transaction_code'),
          total_amount: searchParams.get('total_amount'),
          transaction_uuid: searchParams.get('transaction_uuid'),
          product_code: searchParams.get('product_code'),
          signed_field_names: searchParams.get('signed_field_names'),
          signature: searchParams.get('signature')
        };
        
        if (!paymentData.status || !paymentData.transaction_uuid) {
          setError('Invalid payment response');
          setVerifying(false);
          return;
        }
      }
      
      console.log('=== Payment Data from eSewa ===');
      console.log('Raw payment data:', paymentData);
      console.log('Status:', paymentData.status);
      console.log('Transaction UUID:', paymentData.transaction_uuid);
      console.log('Transaction Code:', paymentData.transaction_code);
      console.log('Total Amount:', paymentData.total_amount);
      console.log('Signature:', paymentData.signature);
      console.log('================================');
      
      // Call backend to verify and upgrade user
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `${API_URL}/payment/verify`,
        { paymentData: paymentData },
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      );
      
      if (response.data.success) {
        setVerified(true);
      } else {
        setError(response.data.message || 'Payment verification failed');
      }
    } catch (err) {
      console.error('Verification error:', err);
      setError(err.response?.data?.message || 'Failed to verify payment');
    } finally {
      setVerifying(false);
    }
  };
  
  return (
    <div className="min-h-screen bg-primary-50 flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-md w-full bg-white rounded-2xl shadow-2xl p-8"
      >
        {verifying ? (
          <div className="text-center">
            <FaSpinner className="animate-spin text-6xl text-primary-600 mx-auto mb-6" />
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Verifying Payment
            </h2>
            <p className="text-gray-600">
              Please wait while we confirm your payment with eSewa...
            </p>
          </div>
        ) : verified ? (
          <div className="text-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', duration: 0.5 }}
            >
              <FaCheckCircle className="text-7xl text-green-500 mx-auto mb-6" />
            </motion.div>
            
            <h2 className="text-3xl font-bold text-gray-800 mb-4">
              Payment Successful!
            </h2>
            
            <p className="text-gray-600 mb-6">
              Thank you for upgrading to Premium. Your payment has been confirmed and your account has been upgraded!
            </p>
            
            <div className="space-y-3">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => {
                  navigate('/profile');
                  window.location.reload(); // Reload to update user context
                }}
                className="w-full btn-primary py-3"
              >
                Go to Profile
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate('/')}
                className="w-full btn-secondary py-3"
              >
                Back to Home
              </motion.button>
            </div>
          </div>
        ) : (
          <div className="text-center">
            <div className="text-6xl mb-6">❌</div>
            
            <h2 className="text-3xl font-bold text-gray-800 mb-4">
              Payment Verification Failed
            </h2>
            
            <p className="text-gray-600 mb-6">
              {error || 'We could not verify your payment. Please contact support if the amount was deducted.'}
            </p>
            
            <div className="space-y-3">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate('/payment')}
                className="w-full btn-primary py-3"
              >
                Try Again
              </motion.button>
              
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate('/')}
                className="w-full btn-secondary py-3"
              >
                Back to Home
              </motion.button>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default PaymentSuccess;
