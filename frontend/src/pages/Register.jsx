import { useState, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import { register as registerUser } from '../services/authService';
import { FaUser, FaEnvelope, FaLock, FaEye, FaEyeSlash, FaCheckCircle, FaTimes, FaCheck } from 'react-icons/fa';
import logoImage from '../images/Logo.png';

const Register = () => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const [passwordFocused, setPasswordFocused] = useState(false);
  const { login: authLogin } = useAuth();
  const navigate = useNavigate();

  // Password validation rules
  const passwordValidation = useMemo(() => {
    const password = formData.password;
    return {
      minLength: password.length >= 8,
      hasUppercase: /[A-Z]/.test(password),
      hasLowercase: /[a-z]/.test(password),
      hasNumber: /[0-9]/.test(password),
      hasSpecialChar: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password),
      passwordsMatch: formData.password && formData.password === formData.confirmPassword,
    };
  }, [formData.password, formData.confirmPassword]);

  // Calculate password strength
  const passwordStrength = useMemo(() => {
    const checks = Object.values(passwordValidation);
    const passed = checks.filter(Boolean).length;
    const total = checks.length - 1; // Exclude passwordsMatch from strength calculation
    const percentage = (passed / total) * 100;
    
    if (percentage === 0) return { label: '', color: 'bg-gray-200', percentage: 0 };
    if (percentage <= 40) return { label: 'Weak', color: 'bg-red-500', percentage };
    if (percentage <= 60) return { label: 'Fair', color: 'bg-orange-500', percentage };
    if (percentage <= 80) return { label: 'Good', color: 'bg-yellow-500', percentage };
    return { label: 'Strong', color: 'bg-green-500', percentage };
  }, [passwordValidation]);

  const isPasswordValid = useMemo(() => {
    return (
      passwordValidation.minLength &&
      passwordValidation.hasUppercase &&
      passwordValidation.hasLowercase &&
      passwordValidation.hasNumber &&
      passwordValidation.hasSpecialChar
    );
  }, [passwordValidation]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Validation
    if (formData.username.length < 3) {
      setError('Username must be at least 3 characters');
      return;
    }

    if (!isPasswordValid) {
      setError('Please ensure your password meets all requirements');
      return;
    }

    if (!passwordValidation.passwordsMatch) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const response = await registerUser(formData.username, formData.email, formData.password);
      
      if (response.success && response.data) {
        // Store auth data
        localStorage.setItem('token', response.data.token);
        localStorage.setItem('user', JSON.stringify({
          id: response.data.userId,
          username: response.data.username,
          email: response.data.email,
          role: response.data.role
        }));

        setSuccess(true);
        
        // Navigate to home after brief delay
        setTimeout(() => {
          navigate('/');
          window.location.reload(); // Reload to update auth context
        }, 2000);
      } else {
        setError(response.message || 'Registration failed');
      }
    } catch (err) {
      setError(err.response?.data?.message || err.message || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-primary-50 flex items-center justify-center px-4 py-12">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-md"
        >
          <div className="bg-white rounded-lg shadow-lg p-8 text-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring" }}
              className="inline-block p-4 bg-green-100 rounded-full mb-6"
            >
              <FaCheckCircle className="text-5xl text-green-600" />
            </motion.div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Registration Successful!
            </h2>
            <p className="text-gray-600 mb-6">
              Welcome to Lipika OCR! You get 10 free scans to start with.
            </p>
            <div className="animate-pulse text-primary-600 font-semibold">
              Redirecting to home...
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-primary-50 flex items-center justify-center px-4 py-12">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="bg-white rounded-lg shadow-lg p-8">
          <div className="text-center mb-8">
            <img 
              src={logoImage} 
              alt="Lipika Logo" 
              className="h-20 w-auto mx-auto mb-4"
            />
            <h1 className="text-2xl font-bold text-gray-900 mb-2">
              Create Your Account
            </h1>
            <p className="text-gray-600 text-sm">
              Get 10 free OCR scans to start with!
            </p>
          </div>

          {error && (
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm"
            >
              {error}
            </motion.div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Username
              </label>
              <div className="relative">
                <FaUser className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="text"
                  name="username"
                  value={formData.username}
                  onChange={handleChange}
                  required
                  minLength={3}
                  className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-colors"
                  placeholder="Choose a username"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email
              </label>
              <div className="relative">
                <FaEnvelope className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-colors"
                  placeholder="Enter your email"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <FaLock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  onFocus={() => setPasswordFocused(true)}
                  required
                  className="w-full pl-10 pr-10 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-colors"
                  placeholder="Create a strong password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {showPassword ? <FaEyeSlash /> : <FaEye />}
                </button>
              </div>

              {/* Password Strength Bar */}
              <AnimatePresence>
                {formData.password && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-2"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-gray-600">Password Strength</span>
                      <span className={`text-xs font-semibold ${
                        passwordStrength.label === 'Weak' ? 'text-red-600' :
                        passwordStrength.label === 'Fair' ? 'text-orange-600' :
                        passwordStrength.label === 'Good' ? 'text-yellow-600' :
                        'text-green-600'
                      }`}>
                        {passwordStrength.label}
                      </span>
                    </div>
                    <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${passwordStrength.percentage}%` }}
                        transition={{ duration: 0.3 }}
                        className={`h-full ${passwordStrength.color} transition-colors duration-300`}
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Password Requirements */}
              <AnimatePresence>
                {(passwordFocused || formData.password) && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-3 p-3 bg-gray-50 rounded-lg border border-gray-200"
                  >
                    <p className="text-xs font-semibold text-gray-700 mb-2">Password must contain:</p>
                    <div className="space-y-1.5">
                      <ValidationItem 
                        isValid={passwordValidation.minLength}
                        text="At least 8 characters"
                      />
                      <ValidationItem 
                        isValid={passwordValidation.hasUppercase}
                        text="One uppercase letter (A-Z)"
                      />
                      <ValidationItem 
                        isValid={passwordValidation.hasLowercase}
                        text="One lowercase letter (a-z)"
                      />
                      <ValidationItem 
                        isValid={passwordValidation.hasNumber}
                        text="One number (0-9)"
                      />
                      <ValidationItem 
                        isValid={passwordValidation.hasSpecialChar}
                        text="One special character (!@#$%^&*)"
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <FaLock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  required
                  className={`w-full pl-10 pr-10 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none transition-colors ${
                    formData.confirmPassword && (passwordValidation.passwordsMatch ? 'border-green-500' : 'border-red-500')
                  }`}
                  placeholder="Confirm your password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  {showConfirmPassword ? <FaEyeSlash /> : <FaEye />}
                </button>
              </div>
              
              {/* Password Match Indicator */}
              <AnimatePresence>
                {formData.confirmPassword && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="mt-2"
                  >
                    {passwordValidation.passwordsMatch ? (
                      <div className="flex items-center gap-2 text-sm text-green-600">
                        <FaCheck className="text-xs" />
                        <span>Passwords match</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-sm text-red-600">
                        <FaTimes className="text-xs" />
                        <span>Passwords do not match</span>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <button
              type="submit"
              disabled={loading || !isPasswordValid || !passwordValidation.passwordsMatch || formData.username.length < 3}
              className={`btn-primary btn-lg w-full focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${
                (loading || !isPasswordValid || !passwordValidation.passwordsMatch || formData.username.length < 3) ? 'btn-disabled' : ''
              }`}
            >
              {loading ? 'Creating account...' : 'Create Account'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              Already have an account?{' '}
              <Link 
                to="/login" 
                className="text-primary-600 hover:text-primary-700 font-semibold transition-colors"
              >
                Login here
              </Link>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

// Validation Item Component
const ValidationItem = ({ isValid, text }) => (
  <motion.div
    initial={{ opacity: 0, x: -10 }}
    animate={{ opacity: 1, x: 0 }}
    className="flex items-center gap-2"
  >
    <div className={`flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center ${
      isValid ? 'bg-green-500' : 'bg-gray-300'
    } transition-colors duration-300`}>
      {isValid ? (
        <FaCheck className="text-white text-xs" />
      ) : (
        <FaTimes className="text-gray-500 text-xs" />
      )}
    </div>
    <span className={`text-xs ${isValid ? 'text-green-700 font-medium' : 'text-gray-600'} transition-colors duration-300`}>
      {text}
    </span>
  </motion.div>
);

export default Register;

