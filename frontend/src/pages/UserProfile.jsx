import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaUser, FaEnvelope, FaClock, FaChartLine, FaCrown, FaSignOutAlt } from 'react-icons/fa';
import { getUserProfile } from '../services/authService';
import { useAuth } from '../context/AuthContext';
import Header from '../components/Header';
import Footer from '../components/Footer';

const UserProfile = () => {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isAuthenticated()) {
      navigate('/login');
      return;
    }

    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await getUserProfile(token);
      
      if (response.success) {
        setProfile(response.data);
      } else {
        setError('Failed to load profile');
      }
    } catch (err) {
      setError('Failed to load profile');
      if (err.response?.status === 401) {
        logout();
        navigate('/login');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const handleUpgradeToPremium = () => {
    // Placeholder for payment integration
    alert('Payment integration coming soon! This will redirect to a payment gateway.');
  };

  const calculateUsagePercentage = () => {
    if (!profile || profile.isPremium) return 0;
    return (profile.usageCount / profile.usageLimit) * 100;
  };

  const getRemainingScans = () => {
    if (!profile) return 0;
    if (profile.isPremium) return 'Unlimited';
    return Math.max(0, profile.usageLimit - profile.usageCount);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-primary-50 flex flex-col">
        <Header />
        <div className="flex-grow flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-600 border-t-transparent"></div>
        </div>
        <Footer />
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="min-h-screen bg-primary-50 flex flex-col">
        <Header />
        <div className="flex-grow flex items-center justify-center px-4">
          <div className="text-center">
            <p className="text-red-600 mb-4">{error || 'Failed to load profile'}</p>
            <button 
              onClick={() => navigate('/')}
              className="btn-primary"
            >
              Go Home
            </button>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-primary-50 flex flex-col">
      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-8 max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Profile Header */}
          <div className="card">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-4">
                <div className="w-20 h-20 bg-primary-600 rounded-full flex items-center justify-center">
                  <FaUser className="text-3xl text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">{profile.username}</h1>
                  <p className="text-gray-600 flex items-center gap-2 mt-1">
                    <FaEnvelope className="text-sm" />
                    {profile.email}
                  </p>
                  {profile.isPremium && (
                    <span className="inline-flex items-center gap-1 mt-2 px-3 py-1 bg-gradient-to-r from-yellow-400 to-orange-500 text-white text-sm font-semibold rounded-full">
                      <FaCrown />
                      Premium Member
                    </span>
                  )}
                </div>
              </div>
              
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <FaSignOutAlt />
                Logout
              </button>
            </div>
          </div>

          {/* Usage Stats */}
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <FaChartLine className="text-primary-600" />
              Usage Statistics
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-primary-50 rounded-lg p-4">
                <p className="text-sm text-gray-600 mb-1">Total Scans</p>
                <p className="text-3xl font-bold text-primary-600">{profile.usageCount}</p>
              </div>
              
              <div className="bg-secondary-50 rounded-lg p-4">
                <p className="text-sm text-gray-600 mb-1">Remaining Scans</p>
                <p className="text-3xl font-bold text-secondary-600">{getRemainingScans()}</p>
              </div>
              
              <div className={`rounded-lg p-4 ${profile.isPremium ? 'bg-gradient-to-r from-yellow-100 to-orange-100' : 'bg-gray-50'}`}>
                <p className="text-sm text-gray-600 mb-1">Status</p>
                <p className="text-xl font-bold text-gray-900">
                  {profile.isPremium ? 'Premium' : 'Free'}
                </p>
              </div>
            </div>

            {/* Usage Progress Bar */}
            {!profile.isPremium && (
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>Usage Limit</span>
                  <span>{profile.usageCount} / {profile.usageLimit}</span>
                </div>
                <div className="w-full h-4 bg-gray-200 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${calculateUsagePercentage()}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    className={`h-full rounded-full ${
                      calculateUsagePercentage() >= 100
                        ? 'bg-red-500'
                        : calculateUsagePercentage() >= 80
                        ? 'bg-orange-500'
                        : 'bg-primary-600'
                    }`}
                  />
                </div>
                
                {profile.usageCount >= profile.usageLimit && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg"
                  >
                    <p className="text-red-700 font-semibold mb-2">
                      ⚠️ You've reached your free scan limit!
                    </p>
                    <p className="text-sm text-red-600 mb-3">
                      Upgrade to Premium for unlimited scans and additional features.
                    </p>
                    <button
                      onClick={handleUpgradeToPremium}
                      className="w-full btn-primary bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600"
                    >
                      <FaCrown className="inline mr-2" />
                      Upgrade to Premium
                    </button>
                  </motion.div>
                )}
              </div>
            )}
          </div>

          {/* Account Details */}
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <FaClock className="text-primary-600" />
              Account Details
            </h2>
            
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-gray-200">
                <span className="text-gray-600">Role</span>
                <span className="font-semibold text-gray-900">{profile.role}</span>
              </div>
              
              <div className="flex justify-between py-2 border-b border-gray-200">
                <span className="text-gray-600">Member Since</span>
                <span className="font-semibold text-gray-900">
                  {new Date(profile.createdAt).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </span>
              </div>
              
              {profile.lastLogin && (
                <div className="flex justify-between py-2 border-b border-gray-200">
                  <span className="text-gray-600">Last Login</span>
                  <span className="font-semibold text-gray-900">
                    {new Date(profile.lastLogin).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                </div>
              )}
              
              {profile.isPremium && profile.premiumUntil && (
                <div className="flex justify-between py-2">
                  <span className="text-gray-600">Premium Until</span>
                  <span className="font-semibold text-yellow-600">
                    {new Date(profile.premiumUntil).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    })}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Upgrade Section for Free Users */}
          {!profile.isPremium && profile.usageCount < profile.usageLimit && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="card bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-200"
            >
              <div className="flex items-start gap-4">
                <FaCrown className="text-4xl text-yellow-500 flex-shrink-0" />
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-gray-900 mb-2">
                    Upgrade to Premium
                  </h3>
                  <p className="text-gray-700 mb-4">
                    Get unlimited OCR scans, priority support, and access to premium features!
                  </p>
                  <button
                    onClick={handleUpgradeToPremium}
                    className="btn-primary bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600"
                  >
                    Learn More
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </motion.div>
      </main>
      
      <Footer />
    </div>
  );
};

export default UserProfile;

