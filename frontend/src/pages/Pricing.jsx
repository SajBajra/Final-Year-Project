import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  FaCheckCircle, 
  FaRocket, 
  FaClock,
  FaInfinity,
  FaHeadset,
  FaStar
} from 'react-icons/fa';
import { useAuth } from '../context/AuthContext';

const Pricing = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  const [billingCycle, setBillingCycle] = useState('monthly');
  
  const plans = [
    {
      name: 'Free',
      price: 0,
      period: 'Forever',
      description: 'Perfect for trying out Lipika OCR',
      features: [
        { text: '5 OCR scans per day', icon: FaClock },
        { text: 'Basic Ranjana script recognition', icon: FaCheckCircle },
        { text: 'Standard processing speed', icon: FaCheckCircle },
        { text: 'Community support', icon: FaCheckCircle }
      ],
      cta: 'Get Started',
      popular: false,
      color: 'gray'
    },
    {
      name: 'Premium',
      price: billingCycle === 'monthly' ? 100 : 1000,
      period: billingCycle === 'monthly' ? 'month' : 'year',
      originalPrice: billingCycle === 'yearly' ? 1200 : null,
      description: 'For power users and professionals',
      features: [
        { text: 'Unlimited OCR scans', icon: FaInfinity },
        { text: 'Advanced character recognition', icon: FaRocket },
        { text: 'Priority processing', icon: FaStar },
        { text: 'Email support', icon: FaHeadset },
        { text: 'No daily limits', icon: FaCheckCircle },
        { text: 'Export to multiple formats', icon: FaCheckCircle }
      ],
      cta: 'Upgrade Now',
      popular: true,
      color: 'primary'
    }
  ];

  const handleSelectPlan = (plan) => {
    if (plan.name === 'Free') {
      if (!isAuthenticated()) {
        navigate('/register');
      } else {
        navigate('/');
      }
    } else {
      if (!isAuthenticated()) {
        navigate('/login', { state: { redirectTo: '/payment' } });
      } else {
        navigate('/payment');
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden py-16 sm:py-20 lg:py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <h1 
              className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6"
              style={{ fontFamily: 'Poppins, sans-serif' }}
            >
              Simple, Transparent Pricing
            </h1>

            {/* Billing Toggle */}
            <div className="flex items-center justify-center gap-4 mb-8">
              <span className={`text-sm font-medium ${billingCycle === 'monthly' ? 'text-gray-900' : 'text-gray-500'}`}>
                Monthly
              </span>
              <motion.button
                whileTap={{ scale: 0.95 }}
                onClick={() => setBillingCycle(billingCycle === 'monthly' ? 'yearly' : 'monthly')}
                className={`relative w-16 h-8 rounded-full transition-colors duration-200 ${
                  billingCycle === 'yearly' ? 'bg-primary-600' : 'bg-gray-300'
                }`}
              >
                <motion.div
                  layout
                  className="absolute top-1 left-1 w-6 h-6 bg-white rounded-full shadow-md"
                  animate={{ x: billingCycle === 'yearly' ? 32 : 0 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                />
              </motion.button>
              <span className={`text-sm font-medium ${billingCycle === 'yearly' ? 'text-gray-900' : 'text-gray-500'}`}>
                Yearly
                <span className="ml-2 inline-block px-2 py-0.5 text-xs font-semibold text-white bg-green-500 rounded-full">
                  Save 17%
                </span>
              </span>
            </div>
          </motion.div>

          {/* Pricing Cards */}
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {plans.map((plan, index) => (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className={`relative bg-white rounded-2xl shadow-xl overflow-hidden ${
                  plan.popular ? 'border-4 border-primary-500 transform scale-105' : 'border-2 border-gray-200'
                }`}
              >
                {plan.popular && (
                  <div className="absolute top-0 right-0 bg-primary-500 text-white px-4 py-1 text-sm font-semibold rounded-bl-lg">
                    Most Popular
                  </div>
                )}

                <div className="p-8">
                  {/* Plan Header */}
                  <div className="text-center mb-6">
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      {plan.name}
                    </h3>
                    <p className="text-gray-600 text-sm mb-4">
                      {plan.description}
                    </p>
                    
                    <div className="flex items-baseline justify-center gap-2">
                      <span className="text-5xl font-extrabold text-gray-900">
                        NPR {plan.price}
                      </span>
                      <span className="text-gray-600">
                        / {plan.period}
                      </span>
                    </div>
                    
                    {plan.originalPrice && (
                      <p className="text-sm text-gray-500 mt-2">
                        <span className="line-through">NPR {plan.originalPrice}</span>
                        <span className="ml-2 text-green-600 font-semibold">
                          Save NPR {plan.originalPrice - plan.price}
                        </span>
                      </p>
                    )}
                  </div>

                  {/* Features */}
                  <ul className="space-y-4 mb-8">
                    {plan.features.map((feature, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <feature.icon className={`flex-shrink-0 w-5 h-5 mt-0.5 ${
                          plan.color === 'primary' ? 'text-primary-600' : 'text-gray-500'
                        }`} />
                        <span className="text-gray-700">{feature.text}</span>
                      </li>
                    ))}
                  </ul>

                  {/* CTA Button */}
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleSelectPlan(plan)}
                    className={`w-full py-4 rounded-xl font-semibold text-lg transition-all duration-200 ${
                      plan.popular
                        ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg hover:shadow-xl'
                        : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
                    }`}
                  >
                    {plan.cta}
                  </motion.button>
                </div>
              </motion.div>
            ))}
          </div>

        </div>
      </div>
    </div>
  );
};

export default Pricing;
