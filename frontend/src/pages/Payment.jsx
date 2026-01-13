import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaCreditCard, FaShoppingCart, FaCheckCircle } from 'react-icons/fa';
import { initiatePayment } from '../services/paymentService';
import { useAuth } from '../context/AuthContext';

const Payment = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const formRef = useRef(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPlan, setSelectedPlan] = useState(null);
  
  const plans = [
    {
      id: 'premium_monthly',
      name: 'Premium Monthly',
      price: 500,
      description: 'Unlimited OCR scans for 30 days',
      features: [
        'Unlimited OCR scans',
        'Priority processing',
        'Advanced features',
        'Email support'
      ],
      productCode: 'PREMIUM_MONTHLY'
    },
    {
      id: 'premium_yearly',
      name: 'Premium Yearly',
      price: 5000,
      description: 'Unlimited OCR scans for 365 days',
      features: [
        'Unlimited OCR scans',
        'Priority processing',
        'Advanced features',
        'Email support',
        'Save 17% compared to monthly'
      ],
      productCode: 'PREMIUM_YEARLY',
      badge: 'Best Value'
    }
  ];
  
  const handlePayment = async (plan) => {
    if (!user) {
      navigate('/login');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSelectedPlan(plan);
    
    try {
      const token = localStorage.getItem('token');
      
      const paymentData = {
        amount: plan.price,
        productName: plan.name,
        productCode: plan.productCode
      };
      
      const response = await initiatePayment(paymentData, token);
      
      if (response.success && response.data) {
        // Create a form and submit to eSewa
        submitToEsewa(response.data);
      } else {
        throw new Error(response.message || 'Failed to initiate payment');
      }
    } catch (err) {
      console.error('Payment error:', err);
      setError(err.message || 'Failed to initiate payment. Please try again.');
      setLoading(false);
    }
  };
  
  const submitToEsewa = (paymentData) => {
    // Create a hidden form
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = paymentData.paymentUrl;
    
    // Add form fields
    const fields = {
      amount: paymentData.amount,
      tax_amount: paymentData.taxAmount,
      total_amount: paymentData.totalAmount,
      transaction_uuid: paymentData.transactionUuid,
      product_code: paymentData.productCode,
      product_service_charge: paymentData.productServiceCharge,
      product_delivery_charge: paymentData.productDeliveryCharge,
      success_url: window.location.origin + '/payment/success',
      failure_url: window.location.origin + '/payment/failure',
      signed_field_names: paymentData.signedFieldNames,
      signature: paymentData.signature
    };
    
    Object.keys(fields).forEach(key => {
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = key;
      input.value = fields[key];
      form.appendChild(input);
    });
    
    document.body.appendChild(form);
    form.submit();
  };
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-secondary-50 py-12 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-4">
            Upgrade to Premium
          </h1>
          <p className="text-lg text-gray-600">
            Choose a plan and unlock unlimited OCR capabilities
          </p>
        </motion.div>
        
        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl mx-auto mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800"
          >
            {error}
          </motion.div>
        )}
        
        {/* Pricing Plans */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`relative bg-white rounded-2xl shadow-xl p-8 ${
                plan.badge ? 'border-4 border-primary-500' : ''
              }`}
            >
              {plan.badge && (
                <div className="absolute top-0 right-8 transform -translate-y-1/2">
                  <span className="bg-primary-500 text-white px-4 py-1 rounded-full text-sm font-semibold">
                    {plan.badge}
                  </span>
                </div>
              )}
              
              <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-gray-800 mb-2">
                  {plan.name}
                </h3>
                <p className="text-gray-600 mb-4">{plan.description}</p>
                <div className="flex items-center justify-center">
                  <span className="text-5xl font-bold text-primary-600">
                    NPR {plan.price}
                  </span>
                </div>
              </div>
              
              <ul className="space-y-3 mb-8">
                {plan.features.map((feature, idx) => (
                  <li key={idx} className="flex items-center text-gray-700">
                    <FaCheckCircle className="text-green-500 mr-3 flex-shrink-0" />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
              
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handlePayment(plan)}
                disabled={loading && selectedPlan?.id === plan.id}
                className={`w-full py-4 rounded-xl font-semibold text-lg transition-all duration-200 ${
                  plan.badge
                    ? 'bg-primary-600 text-white hover:bg-primary-700'
                    : 'bg-gray-100 text-gray-800 hover:bg-gray-200'
                } disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2`}
              >
                {loading && selectedPlan?.id === plan.id ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <FaCreditCard />
                    <span>Pay with eSewa</span>
                  </>
                )}
              </motion.button>
            </motion.div>
          ))}
        </div>
        
        {/* eSewa Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-center text-gray-600"
        >
          <p className="mb-2">Secure payment powered by</p>
          <div className="flex items-center justify-center space-x-2">
            <FaCreditCard className="text-2xl text-primary-600" />
            <span className="text-2xl font-bold text-primary-600">eSewa</span>
          </div>
          <p className="mt-4 text-sm">
            You will be redirected to eSewa's secure payment gateway
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default Payment;
