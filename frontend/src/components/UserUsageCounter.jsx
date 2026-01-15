import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FaCheckCircle, FaExclamationTriangle, FaCrown } from 'react-icons/fa';
import { useAuth } from '../context/AuthContext';
import { getUserProfile } from '../services/authService';

const UserUsageCounter = ({ ocrResult }) => {
  const [usageData, setUsageData] = useState(null);
  const [loading, setLoading] = useState(true);
  const { token, isAuthenticated, isAdmin, isPremium } = useAuth();
  const navigate = useNavigate();

  const fetchUsageData = async () => {
    if (!isAuthenticated() || !token) {
      setLoading(false);
      return;
    }

    try {
      // Fetch user profile to get usage counts
      const response = await getUserProfile(token);
      if (response.success && response.data) {
        const profile = response.data;
        // Only show counter for free users (not premium or admin)
        if (!profile.isPremium && profile.role !== 'ADMIN') {
          const remaining = Math.max(0, profile.usageLimit - profile.usageCount);
          setUsageData({
            remaining: remaining,
            used: profile.usageCount || 0,
            limit: profile.usageLimit || 10,
            hasReachedLimit: remaining === 0
          });
        } else {
          // Premium users or admins don't need the counter
          setUsageData(null);
        }
      }
    } catch (error) {
      console.error('Error fetching usage data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Fetch usage data on mount and when token changes
  useEffect(() => {
    fetchUsageData();
  }, [token, isAuthenticated]);

  // Update when OCR result changes (new scan completed)
  useEffect(() => {
    if (ocrResult?.trialInfo) {
      setUsageData({
        remaining: ocrResult.trialInfo.remainingTrials,
        used: ocrResult.trialInfo.usedTrials,
        limit: ocrResult.trialInfo.maxTrials,
        hasReachedLimit: ocrResult.trialInfo.requiresLogin
      });
    }
  }, [ocrResult]);

  // Don't show usage counter for admins or premium users
  if (isAdmin() || isPremium()) {
    return null;
  }

  // Show loading state with a placeholder
  if (!isAuthenticated() || (loading && !usageData)) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6 p-4 rounded-xl shadow-md border-2 bg-gray-50 border-gray-200"
      >
        <div className="flex items-center gap-2">
          <div className="animate-pulse bg-gray-300 h-4 w-32 rounded"></div>
        </div>
      </motion.div>
    );
  }

  // If no usage data available, try to show default values
  if (!usageData) {
    // Still show counter with default values for free users
    return (
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6 p-4 rounded-xl shadow-md border-2 bg-primary-50 border-primary-300"
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <FaCheckCircle className="text-primary-600 text-lg" />
            <span className="text-base font-bold text-primary-900">
              OCR Scans Remaining: <span className="text-xl">Loading...</span>
            </span>
          </div>
        </div>
      </motion.div>
    );
  }

  const { remaining, used, limit, hasReachedLimit } = usageData;

  // If user has reached limit
  if (hasReachedLimit) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6 p-4 bg-orange-50 border-2 border-orange-300 rounded-xl shadow-md"
      >
        <div className="flex items-start gap-3">
          <FaExclamationTriangle className="text-orange-600 text-2xl mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="font-bold text-orange-900 mb-1 text-lg">Usage Limit Reached</h3>
            <p className="text-sm text-orange-800 mb-3">
              You've used all {limit} free OCR scans. Upgrade to premium for unlimited access!
            </p>
            <button 
              onClick={() => navigate('/pricing')}
              className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white px-5 py-2.5 rounded-lg font-semibold hover:from-yellow-600 hover:to-orange-600 transition-all shadow-md hover:shadow-lg flex items-center gap-2"
            >
              <FaCrown />
              <span>Upgrade to Premium</span>
            </button>
          </div>
        </div>
      </motion.div>
    );
  }

  // Show usage counter if we have the data
  if (remaining !== undefined && limit !== undefined) {
    const percentage = (remaining / limit) * 100;
    const isLow = remaining <= 3;

    return (
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`mb-6 p-4 rounded-xl shadow-md border-2 ${
          isLow 
            ? 'bg-orange-50 border-orange-300' 
            : 'bg-primary-50 border-primary-300'
        }`}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {isLow ? (
              <FaExclamationTriangle className="text-orange-600 text-lg" />
            ) : (
              <FaCheckCircle className="text-primary-600 text-lg" />
            )}
            <span className={`text-base font-bold ${
              isLow ? 'text-orange-900' : 'text-primary-900'
            }`}>
              OCR Scans Remaining: <span className="text-xl">{remaining}/{limit}</span>
            </span>
          </div>
          <div className={`text-sm font-medium px-3 py-1 rounded-full ${
            isLow 
              ? 'bg-orange-200 text-orange-800' 
              : 'bg-primary-200 text-primary-800'
          }`}>
            {used} used
          </div>
        </div>
        
        {/* Progress Bar */}
        <div className={`w-full rounded-full h-3 ${
          isLow ? 'bg-orange-200' : 'bg-primary-200'
        }`}>
          <motion.div
            initial={{ width: '100%' }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className={`h-3 rounded-full ${
              isLow 
                ? 'bg-gradient-to-r from-orange-500 to-red-500' 
                : 'bg-gradient-to-r from-primary-500 to-primary-600'
            }`}
          />
        </div>
        
        {isLow && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-xs text-orange-700 mt-2 font-medium"
          >
            ⚠️ Running low! Upgrade to premium for unlimited scans.
          </motion.p>
        )}
      </motion.div>
    );
  }

  return null;
};

export default UserUsageCounter;

