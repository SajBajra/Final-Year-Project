import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaUser, FaEnvelope, FaClock, FaChartLine, FaCrown, FaSignOutAlt, FaCheckCircle, FaFire, FaCalendar, FaStar, FaKey } from 'react-icons/fa';
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
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 flex flex-col" style={{ fontFamily: 'Poppins, sans-serif' }}>
      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-8 max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <h1 className="text-3xl md:text-4xl font-black text-gray-900 mb-2" style={{ fontFamily: 'Poppins, sans-serif' }}>My Profile</h1>
          <p className="text-gray-600" style={{ fontFamily: 'Poppins, sans-serif' }}>Manage your account and track your OCR usage</p>
        </motion.div>

        {/* Bento Grid Layout */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6 auto-rows-fr">
          
          {/* Profile Card - Large */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="md:col-span-2 lg:row-span-2 card bg-gradient-to-br from-primary-600 to-primary-700 text-white relative overflow-hidden"
            style={{ fontFamily: 'Poppins, sans-serif' }}
          >
            <div className="relative z-10">
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-center gap-4">
                  <motion.div 
                    whileHover={{ scale: 1.05 }}
                    className="w-20 h-20 bg-white/20 backdrop-blur-sm rounded-2xl flex items-center justify-center border-2 border-white/30"
                  >
                    <FaUser className="text-4xl" />
                  </motion.div>
                  <div>
                    <h2 className="text-2xl md:text-3xl font-bold mb-1">{profile.username}</h2>
                    <p className="text-white/80 flex items-center gap-2 text-sm">
                      <FaEnvelope />
                      {profile.email}
                    </p>
                  </div>
                </div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleLogout}
                  className="p-3 bg-white/10 hover:bg-white/20 backdrop-blur-sm rounded-xl transition-colors border border-white/20"
                >
                  <FaSignOutAlt className="text-xl" />
                </motion.button>
              </div>

              <div className="space-y-3 mb-6">
                {profile.isPremium && (
                  <div className="flex items-center gap-2 bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-4 py-2 rounded-xl font-semibold">
                    <FaCrown className="text-lg" />
                    <span>Premium Member</span>
                  </div>
                )}
                
                <div className="flex items-center gap-2 text-sm text-white/80 bg-white/10 backdrop-blur-sm px-4 py-2 rounded-xl border border-white/20">
                  <FaCheckCircle />
                  <span>Role: {profile.role}</span>
                </div>
                
                <div className="flex items-center gap-2 text-sm text-white/80 bg-white/10 backdrop-blur-sm px-4 py-2 rounded-xl border border-white/20">
                  <FaCalendar />
                  <span>Joined {new Date(profile.createdAt).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</span>
                </div>

                {profile.lastLogin && (
                  <div className="flex items-center gap-2 text-sm text-white/80 bg-white/10 backdrop-blur-sm px-4 py-2 rounded-xl border border-white/20">
                    <FaClock />
                    <span>Last Login: {new Date(profile.lastLogin).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                  </div>
                )}

                {profile.isPremium && profile.premiumUntil && (
                  <div className="flex items-center gap-2 text-sm text-white/80 bg-gradient-to-r from-yellow-400/20 to-orange-500/20 backdrop-blur-sm px-4 py-2 rounded-xl border border-yellow-400/30">
                    <FaCrown />
                    <span>Premium Until: {new Date(profile.premiumUntil).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                  </div>
                )}
              </div>

              {/* Change Password Button */}
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="w-full flex items-center justify-center gap-2 bg-white/10 hover:bg-white/15 backdrop-blur-sm px-4 py-3 rounded-xl border border-white/20 transition-colors text-sm font-medium"
              >
                <FaKey />
                <span>Change Password</span>
              </motion.button>
            </div>
          </motion.div>

          {/* Total Scans Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="card bg-gradient-to-br from-blue-500 to-blue-600 text-white"
            style={{ fontFamily: 'Poppins, sans-serif' }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-white/20 rounded-xl">
                <FaFire className="text-2xl" />
              </div>
              <p className="text-white/90 text-sm font-medium">Total Scans</p>
            </div>
            <p className="text-5xl font-black mb-2">{profile.usageCount}</p>
            <p className="text-white/70 text-xs">OCR operations performed</p>
          </motion.div>

          {/* Remaining Scans Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className={`card ${
              profile.isPremium 
                ? 'bg-gradient-to-br from-yellow-400 to-orange-500' 
                : getRemainingScans() <= 3 
                  ? 'bg-gradient-to-br from-red-500 to-red-600'
                  : 'bg-gradient-to-br from-green-500 to-green-600'
            } text-white`}
            style={{ fontFamily: 'Poppins, sans-serif' }}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-white/20 rounded-xl">
                <FaStar className="text-2xl" />
              </div>
              <p className="text-white/90 text-sm font-medium">Remaining</p>
            </div>
            <p className="text-5xl font-black mb-2">{getRemainingScans()}</p>
            <p className="text-white/70 text-xs">{profile.isPremium ? 'Unlimited access' : 'scans left'}</p>
          </motion.div>

          {/* Usage Chart Card */}
          {!profile.isPremium && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="md:col-span-2 lg:col-span-2 card"
              style={{ fontFamily: 'Poppins, sans-serif' }}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-primary-100 rounded-xl">
                  <FaChartLine className="text-primary-600 text-xl" />
                </div>
                <h3 className="text-lg font-bold text-gray-900">Usage Progress</h3>
              </div>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Current Usage</span>
                  <span className="text-2xl font-bold text-primary-600">
                    {profile.usageCount} / {profile.usageLimit}
                  </span>
                </div>
                
                <div className="relative">
                  <div className="w-full h-6 bg-gray-200 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${calculateUsagePercentage()}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className={`h-full rounded-full ${
                        calculateUsagePercentage() >= 100
                          ? 'bg-gradient-to-r from-red-500 to-red-600'
                          : calculateUsagePercentage() >= 80
                          ? 'bg-gradient-to-r from-orange-500 to-orange-600'
                          : 'bg-gradient-to-r from-primary-500 to-primary-600'
                      }`}
                    />
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-xs font-bold text-white drop-shadow-md">
                      {Math.round(calculateUsagePercentage())}%
                    </span>
                  </div>
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
            </motion.div>
          )}

          {/* Upgrade Section for Free Users */}
          {!profile.isPremium && profile.usageCount < profile.usageLimit && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="md:col-span-2 lg:col-span-4 card bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-200"
              style={{ fontFamily: 'Poppins, sans-serif' }}
            >
              <div className="flex items-start gap-4">
                <div className="p-3 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-2xl flex-shrink-0">
                  <FaCrown className="text-3xl text-white" />
                </div>
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
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default UserProfile;

