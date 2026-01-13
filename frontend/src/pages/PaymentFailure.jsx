import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaTimesCircle } from 'react-icons/fa';

const PaymentFailure = () => {
  const navigate = useNavigate();
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 to-orange-50 flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-md w-full bg-white rounded-2xl shadow-2xl p-8"
      >
        <div className="text-center">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', duration: 0.5 }}
          >
            <FaTimesCircle className="text-7xl text-red-500 mx-auto mb-6" />
          </motion.div>
          
          <h2 className="text-3xl font-bold text-gray-800 mb-4">
            Payment Failed
          </h2>
          
          <p className="text-gray-600 mb-6">
            Your payment could not be processed. This could be due to:
          </p>
          
          <ul className="text-left space-y-2 mb-8 text-gray-700">
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Insufficient balance in your eSewa account</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Payment was cancelled</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Network connectivity issues</span>
            </li>
            <li className="flex items-start">
              <span className="mr-2">•</span>
              <span>Transaction timeout</span>
            </li>
          </ul>
          
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
            <p className="text-sm text-yellow-800">
              <strong>Note:</strong> No amount has been deducted from your account.
            </p>
          </div>
          
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
      </motion.div>
    </div>
  );
};

export default PaymentFailure;
