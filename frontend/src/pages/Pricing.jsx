import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  FaInfinity,
  FaStar,
  FaCheckCircle
} from 'react-icons/fa';
import { useAuth } from '../context/AuthContext';

const Pricing = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();
  
  const plans = [
    {
      name: 'Premium Monthly',
      price: 100,
      period: 'month',
      description: 'Perfect for short-term needs',
      features: [
        { text: 'Unlimited OCR scans', icon: FaInfinity },
        { text: 'Priority processing', icon: FaStar },
        { text: 'No daily limits', icon: FaCheckCircle }
      ],
      cta: 'Get Monthly',
      popular: false,
      color: 'primary'
    },
    {
      name: 'Premium Yearly',
      price: 1000,
      period: 'year',
      originalPrice: 1200,
      description: 'Best value for committed users',
      features: [
        { text: 'Unlimited OCR scans', icon: FaInfinity },
        { text: 'Priority processing', icon: FaStar },
        { text: 'No daily limits', icon: FaCheckCircle },
        { text: 'Save 17% compared to monthly', icon: FaCheckCircle }
      ],
      cta: 'Get Yearly',
      popular: true,
      color: 'primary'
    }
  ];

  const handleSelectPlan = () => {
    if (!isAuthenticated()) {
      navigate('/login', { state: { redirectTo: '/payment' } });
    } else {
      navigate('/payment');
    }
  };

  return (
    <div className="min-h-screen bg-primary-50">
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
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Choose the plan that works best for you. All plans include unlimited OCR scans.
            </p>
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
                        <feature.icon className="flex-shrink-0 w-5 h-5 mt-0.5 text-primary-600" />
                        <span className="text-gray-700">{feature.text}</span>
                      </li>
                    ))}
                  </ul>

                  {/* CTA Button */}
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleSelectPlan}
                    className="w-full py-4 rounded-xl font-semibold text-lg transition-all duration-200 bg-primary-600 text-white hover:bg-primary-700 shadow-lg hover:shadow-xl"
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
