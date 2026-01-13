import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaCreditCard, FaCheckCircle, FaShieldAlt } from 'react-icons/fa';
import CryptoJS from 'crypto-js';
import { useAuth } from '../context/AuthContext';

const Payment = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const formRef = useRef(null);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // eSewa Configuration
  const ESEWA_CONFIG = {
    merchantCode: 'EPAYTEST',
    secretKey: '8gBm/:&EnhH.1/q',
    paymentUrl: 'https://rc-epay.esewa.com.np/api/epay/main/v2/form',
    successUrl: window.location.origin + '/payment/success',
    failureUrl: window.location.origin + '/payment/failure'
  };
  
  const plans = [
    {
      id: 'premium_monthly',
      name: 'Premium Monthly',
      price: 100,
      description: 'Unlimited OCR scans for 30 days',
      features: [
        'Unlimited OCR scans',
        'Priority processing',
        'Advanced features',
        'Email support'
      ]
    },
    {
      id: 'premium_yearly',
      name: 'Premium Yearly',
      price: 1000,
      description: 'Unlimited OCR scans for 365 days',
      features: [
        'Unlimited OCR scans',
        'Priority processing',
        'Advanced features',
        'Email support',
        'Save 17% compared to monthly'
      ],
      badge: 'Best Value'
    }
  ];
  
  const generateEsewaSignature = (totalAmount, transactionUuid, productCode) => {
    // Message format required by eSewa
    const message = `total_amount=${totalAmount},transaction_uuid=${transactionUuid},product_code=${productCode}`;
    
    // Generate HMAC-SHA256 signature using crypto.js
    const hash = CryptoJS.HmacSHA256(message, ESEWA_CONFIG.secretKey);
    const signature = CryptoJS.enc.Base64.stringify(hash);
    
    console.log('=== eSewa Signature Generation ===');
    console.log('Message:', message);
    console.log('Signature:', signature);
    console.log('==================================');
    
    return signature;
  };
  
  const handlePayment = (plan) => {
    if (!user) {
      navigate('/login');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Generate unique transaction UUID
      const transactionUuid = 'TXN-' + Date.now() + '-' + Math.random().toString(36).substring(7);
      
      const amount = plan.price;
      const taxAmount = 0;
      const serviceCharge = 0;
      const deliveryCharge = 0;
      const totalAmount = amount + taxAmount + serviceCharge + deliveryCharge;
      
      // Generate signature using crypto.js
      const signature = generateEsewaSignature(
        totalAmount,
        transactionUuid,
        ESEWA_CONFIG.merchantCode
      );
      
      // Create form and submit to eSewa
      const form = document.createElement('form');
      form.method = 'POST';
      form.action = ESEWA_CONFIG.paymentUrl;
      
      const formFields = {
        amount: amount,
        tax_amount: taxAmount,
        total_amount: totalAmount,
        transaction_uuid: transactionUuid,
        product_code: ESEWA_CONFIG.merchantCode,
        product_service_charge: serviceCharge,
        product_delivery_charge: deliveryCharge,
        success_url: ESEWA_CONFIG.successUrl,
        failure_url: ESEWA_CONFIG.failureUrl,
        signed_field_names: 'total_amount,transaction_uuid,product_code',
        signature: signature
      };
      
      // Add hidden fields to form
      Object.keys(formFields).forEach(key => {
        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = key;
        input.value = formFields[key];
        form.appendChild(input);
      });
      
      // Add form to page and submit
      document.body.appendChild(form);
      form.submit();
      
    } catch (err) {
      console.error('Payment error:', err);
      setError('Failed to initiate payment. Please try again.');
      setLoading(false);
    }
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
            Secure payment via eSewa with crypto.js signature verification
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
                <div className="flex items-baseline justify-center mb-2">
                  <span className="text-5xl font-bold text-primary-600">
                    NPR {plan.price}
                  </span>
                </div>
                <p className="text-gray-600">
                  {plan.description}
                </p>
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
                disabled={loading}
                className={`w-full py-4 rounded-xl font-semibold text-lg transition-all duration-200 ${
                  plan.badge
                    ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg hover:shadow-xl'
                    : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <span className="flex items-center justify-center">
                  <FaCreditCard className="mr-2" />
                  Pay with eSewa
                </span>
              </motion.button>
            </motion.div>
          ))}
        </div>
        
        {/* Security Info */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="max-w-3xl mx-auto bg-white rounded-xl p-6 shadow-lg"
        >
          <div className="flex items-start gap-4">
            <FaShieldAlt className="text-green-600 text-3xl flex-shrink-0 mt-1" />
            <div>
              <h3 className="font-bold text-gray-800 mb-2">🔐 Secure Payment Gateway</h3>
              <p className="text-gray-600 text-sm mb-2">
                Payment processed through <strong>eSewa</strong> - Nepal's #1 payment gateway. 
                Signature generated using <strong>crypto.js HMAC-SHA256</strong> for enhanced security.
              </p>
              <ul className="text-gray-600 text-sm space-y-1">
                <li>✓ Cryptographic signature verification</li>
                <li>✓ Secure redirect to eSewa payment page</li>
                <li>✓ Payment confirmation and verification</li>
                <li>✓ Test mode - Use eSewa test credentials</li>
              </ul>
              <div className="mt-3 p-3 bg-blue-50 rounded-lg text-xs text-gray-700">
                <strong>Test Credentials:</strong><br/>
                eSewa ID: 9806800001/2/3/4/5<br/>
                Password: Nepal@123<br/>
                MPIN: 1122
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Payment;
