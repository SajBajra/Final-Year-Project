import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaUser, FaEnvelope, FaClock, FaChartLine, FaCrown, FaSignOutAlt, FaCheckCircle, FaFire, FaCalendar, FaStar, FaRocket, FaBolt, FaInfinity, FaShieldAlt, FaHeadset, FaZap } from 'react-icons/fa';
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
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-primary-50/30 to-secondary-50/30 flex flex-col">
      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-8 max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl md:text-5xl font-black text-gray-900 mb-2 bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
            My Profile
          </h1>
          <p className="text-gray-600 text-lg">Manage your account and track your OCR usage</p>
        </motion.div>

        {/* Bento Grid Layout */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 md:gap-6">
          
          {/* Profile Card - Large */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            className="md:col-span-3 lg:col-span-4 card bg-gradient-to-br from-primary-600 via-primary-700 to-secondary-600 text-white relative overflow-hidden min-h-[280px]"
          >
            {/* Decorative Elements */}
            <div className="absolute top-0 right-0 w-40 h-40 bg-white/10 rounded-full blur-3xl -mr-20 -mt-20"></div>
            <div className="absolute bottom-0 left-0 w-32 h-32 bg-white/10 rounded-full blur-2xl -ml-16 -mb-16"></div>
            <div className="absolute top-1/2 right-1/4 w-20 h-20 bg-secondary-400/20 rounded-full blur-xl"></div>
            
            <div className="relative z-10 h-full flex flex-col">
              <div className="flex items-start justify-between mb-8">
                <div className="flex items-center gap-4">
                  <motion.div 
                    whileHover={{ scale: 1.05, rotate: 5 }}
                    className="w-24 h-24 bg-white/20 backdrop-blur-md rounded-3xl flex items-center justify-center border-2 border-white/40 shadow-2xl"
                  >
                    <FaUser className="text-5xl drop-shadow-lg" />
                  </motion.div>
                  <div>
                    <h2 className="text-3xl md:text-4xl font-black mb-2 drop-shadow-md">{profile.username}</h2>
                    <p className="text-white/90 flex items-center gap-2 text-base mb-2">
                      <FaEnvelope className="text-lg" />
                      {profile.email}
                    </p>
                    {profile.isPremium && (
                      <motion.div 
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.3, type: "spring" }}
                        className="flex items-center gap-2 bg-gradient-to-r from-yellow-300 to-orange-400 text-gray-900 px-4 py-2 rounded-full font-black shadow-lg"
                      >
                        <FaCrown className="text-xl" />
                        <span>PREMIUM</span>
                      </motion.div>
                    )}
                  </div>
                </div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleLogout}
                  className="p-4 bg-white/10 hover:bg-red-500/80 backdrop-blur-md rounded-2xl transition-all border border-white/30 group"
                >
                  <FaSignOutAlt className="text-2xl group-hover:rotate-12 transition-transform" />
                </motion.button>
              </div>

              <div className="mt-auto grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="flex items-center gap-3 bg-white/10 backdrop-blur-md px-4 py-3 rounded-2xl border border-white/30">
                  <FaCheckCircle className="text-2xl text-green-300" />
                  <div>
                    <p className="text-xs text-white/70">Role</p>
                    <p className="font-bold">{profile.role}</p>
                  </div>
                </div>
                
                <div className="flex items-center gap-3 bg-white/10 backdrop-blur-md px-4 py-3 rounded-2xl border border-white/30">
                  <FaCalendar className="text-2xl text-blue-300" />
                  <div>
                    <p className="text-xs text-white/70">Joined</p>
                    <p className="font-bold">{new Date(profile.createdAt).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}</p>
                  </div>
                </div>

                {profile.lastLogin && (
                  <div className="flex items-center gap-3 bg-white/10 backdrop-blur-md px-4 py-3 rounded-2xl border border-white/30">
                    <FaClock className="text-2xl text-purple-300" />
                    <div>
                      <p className="text-xs text-white/70">Last Login</p>
                      <p className="font-bold text-sm">
                        {new Date(profile.lastLogin).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>

          {/* Total Scans Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="md:col-span-1 lg:col-span-2 card bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 text-white relative overflow-hidden group hover:shadow-2xl transition-shadow min-h-[180px]"
          >
            <motion.div 
              className="absolute top-0 right-0 text-white/10 text-9xl"
              animate={{ rotate: [0, 10, 0] }}
              transition={{ duration: 3, repeat: Infinity }}
              style={{ marginRight: '-1rem', marginTop: '-1rem' }}
            >
              <FaFire />
            </motion.div>
            <div className="relative z-10 h-full flex flex-col justify-between">
              <div>
                <p className="text-white/80 text-sm font-bold mb-1 uppercase tracking-wider">Total Scans</p>
                <motion.p 
                  className="text-6xl md:text-7xl font-black mb-2 drop-shadow-lg"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
                >
                  {profile.usageCount}
                </motion.p>
              </div>
              <div className="flex items-center gap-2 bg-white/20 backdrop-blur-sm px-3 py-2 rounded-xl w-fit">
                <FaZap className="text-yellow-300" />
                <p className="text-white/90 text-xs font-semibold">Operations Performed</p>
              </div>
            </div>
          </motion.div>

          {/* Remaining Scans Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className={`md:col-span-2 lg:col-span-2 card relative overflow-hidden group hover:shadow-2xl transition-shadow min-h-[180px] ${
              profile.isPremium 
                ? 'bg-gradient-to-br from-yellow-400 via-orange-500 to-red-500' 
                : getRemainingScans() <= 3 
                  ? 'bg-gradient-to-br from-red-500 via-red-600 to-pink-600'
                  : 'bg-gradient-to-br from-emerald-500 via-green-600 to-teal-600'
            } text-white`}
          >
            <motion.div 
              className="absolute top-0 right-0 text-white/10 text-9xl"
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              style={{ marginRight: '-1rem', marginTop: '-1rem' }}
            >
              {profile.isPremium ? <FaInfinity /> : <FaStar />}
            </motion.div>
            <div className="relative z-10 h-full flex flex-col justify-between">
              <div>
                <p className="text-white/90 text-sm font-bold mb-1 uppercase tracking-wider">
                  {profile.isPremium ? 'Access Level' : 'Remaining Scans'}
                </p>
                <motion.p 
                  className="text-6xl md:text-7xl font-black mb-2 drop-shadow-lg"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.4, type: "spring", stiffness: 200 }}
                >
                  {getRemainingScans()}
                </motion.p>
              </div>
              <div className="flex items-center gap-2 bg-white/20 backdrop-blur-sm px-3 py-2 rounded-xl w-fit">
                {profile.isPremium ? (
                  <>
                    <FaCrown className="text-yellow-200" />
                    <p className="text-white/90 text-xs font-semibold">Unlimited Premium</p>
                  </>
                ) : getRemainingScans() <= 3 ? (
                  <>
                    <FaBolt className="text-yellow-300 animate-pulse" />
                    <p className="text-white/90 text-xs font-semibold">Running Low!</p>
                  </>
                ) : (
                  <>
                    <FaCheckCircle className="text-green-200" />
                    <p className="text-white/90 text-xs font-semibold">Scans Available</p>
                  </>
                )}
              </div>
            </div>
          </motion.div>

          {/* Usage Chart Card */}
          {!profile.isPremium && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="md:col-span-3 lg:col-span-6 card bg-gradient-to-br from-white to-gray-50"
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-primary-100 rounded-2xl">
                    <FaChartLine className="text-primary-600 text-2xl" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">Usage Progress</h3>
                    <p className="text-sm text-gray-500">Track your OCR consumption</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-3xl font-black text-gray-900">
                    {profile.usageCount}<span className="text-gray-400">/{profile.usageLimit}</span>
                  </p>
                  <p className="text-xs text-gray-500">scans used</p>
                </div>
              </div>
              
              <div className="space-y-3">
                <div className="relative h-8 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${calculateUsagePercentage()}%` }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                    className={`h-full rounded-full relative ${
                      calculateUsagePercentage() >= 100
                        ? 'bg-gradient-to-r from-red-500 via-red-600 to-pink-600'
                        : calculateUsagePercentage() >= 80
                        ? 'bg-gradient-to-r from-orange-500 via-orange-600 to-red-500'
                        : 'bg-gradient-to-r from-primary-500 via-primary-600 to-secondary-600'
                    }`}
                  >
                    <motion.div
                      animate={{ x: [-20, 300], opacity: [0, 1, 0] }}
                      transition={{ duration: 2, repeat: Infinity }}
                      className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                    />
                  </motion.div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-sm font-black text-white drop-shadow-lg">
                      {Math.round(calculateUsagePercentage())}%
                    </span>
                  </div>
                </div>

                {calculateUsagePercentage() >= 80 && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`p-4 rounded-2xl border-2 flex items-center gap-3 ${
                      calculateUsagePercentage() >= 100 
                        ? 'bg-red-50 border-red-200' 
                        : 'bg-orange-50 border-orange-200'
                    }`}
                  >
                    <FaBolt className={`text-3xl ${
                      calculateUsagePercentage() >= 100 ? 'text-red-600' : 'text-orange-600'
                    } animate-pulse`} />
                    <div className="flex-1">
                      <p className={`font-bold text-sm ${
                        calculateUsagePercentage() >= 100 ? 'text-red-700' : 'text-orange-700'
                      }`}>
                        {calculateUsagePercentage() >= 100 
                          ? '⚠️ Usage Limit Reached!' 
                          : '⚠️ Running Low on Scans!'
                        }
                      </p>
                      <p className={`text-xs ${
                        calculateUsagePercentage() >= 100 ? 'text-red-600' : 'text-orange-600'
                      }`}>
                        {calculateUsagePercentage() >= 100 
                          ? 'Upgrade to Premium for unlimited access'
                          : `Only ${getRemainingScans()} scans remaining`
                        }
                      </p>
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          )}

          {/* Premium Upgrade Section */}
          {!profile.isPremium && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="md:col-span-3 lg:col-span-6"
            >
              <div className="card bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white relative overflow-hidden p-8 md:p-10">
                {/* Animated Background Elements */}
                <div className="absolute inset-0 overflow-hidden">
                  <motion.div
                    animate={{ 
                      scale: [1, 1.2, 1],
                      rotate: [0, 180, 360]
                    }}
                    transition={{ duration: 20, repeat: Infinity }}
                    className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-yellow-500/20 to-orange-500/20 rounded-full blur-3xl"
                  />
                  <motion.div
                    animate={{ 
                      scale: [1.2, 1, 1.2],
                      rotate: [360, 180, 0]
                    }}
                    transition={{ duration: 15, repeat: Infinity }}
                    className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-to-tr from-purple-500/20 to-pink-500/20 rounded-full blur-3xl"
                  />
                </div>

                <div className="relative z-10">
                  <div className="flex flex-col lg:flex-row items-start lg:items-center gap-8">
                    {/* Left Section - Icon & Title */}
                    <div className="flex-shrink-0">
                      <motion.div
                        animate={{ 
                          rotate: [0, 5, -5, 0],
                          scale: [1, 1.05, 1]
                        }}
                        transition={{ duration: 3, repeat: Infinity }}
                        className="w-24 h-24 md:w-32 md:h-32 bg-gradient-to-br from-yellow-400 via-orange-500 to-red-500 rounded-3xl flex items-center justify-center shadow-2xl"
                      >
                        <FaCrown className="text-5xl md:text-6xl text-white drop-shadow-lg" />
                      </motion.div>
                    </div>

                    {/* Middle Section - Content */}
                    <div className="flex-1 space-y-6">
                      <div>
                        <motion.div
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.2 }}
                          className="flex items-center gap-2 mb-2"
                        >
                          <FaRocket className="text-yellow-400 text-xl" />
                          <span className="text-yellow-400 text-sm font-bold uppercase tracking-widest">Premium Plan</span>
                        </motion.div>
                        <h3 className="text-3xl md:text-4xl font-black mb-3 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                          Unlock Unlimited Power
                        </h3>
                        <p className="text-gray-300 text-lg">
                          Take your OCR experience to the next level with unlimited scans and exclusive features!
                        </p>
                      </div>

                      {/* Feature Grid */}
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        {[
                          { icon: <FaInfinity />, title: 'Unlimited Scans', desc: 'No limits, scan as much as you need' },
                          { icon: <FaBolt />, title: 'Priority Processing', desc: 'Faster OCR speeds & results' },
                          { icon: <FaHeadset />, title: '24/7 Support', desc: 'Dedicated premium support team' },
                          { icon: <FaShieldAlt />, title: 'Advanced Security', desc: 'Enterprise-grade protection' }
                        ].map((feature, idx) => (
                          <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.3 + idx * 0.1 }}
                            className="flex items-start gap-3 bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/20 hover:bg-white/15 transition-all group"
                          >
                            <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-xl flex items-center justify-center text-white group-hover:scale-110 transition-transform">
                              {feature.icon}
                            </div>
                            <div>
                              <p className="font-bold text-white text-sm">{feature.title}</p>
                              <p className="text-gray-400 text-xs">{feature.desc}</p>
                            </div>
                          </motion.div>
                        ))}
                      </div>

                      {/* Pricing */}
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.7 }}
                        className="flex items-baseline gap-2"
                      >
                        <span className="text-5xl font-black text-white">$9.99</span>
                        <span className="text-gray-400">/month</span>
                        <span className="ml-2 px-3 py-1 bg-green-500/20 border border-green-500/30 rounded-full text-green-400 text-xs font-bold">
                          BEST VALUE
                        </span>
                      </motion.div>
                    </div>

                    {/* Right Section - CTA */}
                    <div className="flex-shrink-0 w-full lg:w-auto">
                      <motion.button
                        whileHover={{ scale: 1.05, y: -5 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleUpgradeToPremium}
                        className="w-full lg:w-auto px-10 py-5 bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500 text-white font-black text-lg rounded-2xl shadow-2xl hover:shadow-yellow-500/50 transition-all flex items-center justify-center gap-3 group"
                      >
                        <FaRocket className="text-2xl group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
                        <span>Upgrade Now</span>
                      </motion.button>
                      <p className="text-center text-gray-400 text-xs mt-3">
                        ✨ 30-day money-back guarantee
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Premium Active Badge */}
          {profile.isPremium && profile.premiumUntil && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
              className="md:col-span-3 lg:col-span-6 card bg-gradient-to-r from-yellow-50 via-orange-50 to-yellow-50 border-2 border-yellow-300"
            >
              <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-2xl flex items-center justify-center">
                    <FaCrown className="text-3xl text-white" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600 font-semibold">Premium Membership Active</p>
                    <p className="text-lg font-black text-gray-900">
                      Valid until {new Date(profile.premiumUntil).toLocaleDateString('en-US', {
                        month: 'long',
                        day: 'numeric',
                        year: 'numeric'
                      })}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2 bg-green-100 px-4 py-2 rounded-full">
                  <FaCheckCircle className="text-green-600" />
                  <span className="font-bold text-green-700">Active</span>
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

