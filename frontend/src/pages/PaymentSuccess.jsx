import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaCheckCircle, FaSpinner } from 'react-icons/fa';
import { verifyPayment } from '../services/paymentService';

const PaymentSuccess = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [verifying, setVerifying] = useState(true);
  const [verified, setVerified] = useState(false);
  const [paymentDetails, setPaymentDetails] = useState(null);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const data = searchParams.get('data');
    
    if (!data) {
      setError('Invalid payment response');
      setVerifying(false);
      return;
    }
    
    // Verify payment
    const verify = async () => {
      try {
        const response = await verifyPayment(data);
        
        if (response.success && response.data) {
          setPaymentDetails(response.data);
          setVerified(response.data.status === 'COMPLETE');
        } else {
          setError(response.message || 'Payment verification failed');
        }
      } catch (err) {
        console.error('Verification error:', err);
        setError(err.message || 'Failed to verify payment');
      } finally {
        setVerifying(false);
      }
    };
    
    verify();
  }, [searchParams]);
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 flex items-center justify-center px-4">
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
              Please wait while we confirm your payment...
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
              Thank you for upgrading to Premium. Your payment has been confirmed.
            </p>
            
            {paymentDetails && (
              <div className="bg-gray-50 rounded-xl p-4 mb-6 text-left">
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Transaction ID:</span>
                    <span className="font-semibold text-gray-800 truncate ml-2">
                      {paymentDetails.transactionUuid}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Amount:</span>
                    <span className="font-semibold text-gray-800">
                      NPR {paymentDetails.totalAmount}
                    </span>
                  </div>
                  {paymentDetails.refId && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Reference ID:</span>
                      <span className="font-semibold text-gray-800">
                        {paymentDetails.refId}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
            
            <div className="space-y-3">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => navigate('/profile')}
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
