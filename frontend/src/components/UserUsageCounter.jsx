import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaCheckCircle, FaExclamationTriangle, FaCrown } from 'react-icons/fa';
import { useAuth } from '../context/AuthContext';
import { checkUsageStatus } from '../services/authService';

const UserUsageCounter = ({ ocrResult }) => {
  const [usageData, setUsageData] = useState(null);
  const [loading, setLoading] = useState(true);
  const { token, isAuthenticated, isAdmin } = useAuth();

  const fetchUsageData = async () => {
    if (!isAuthenticated() || !token) {
      setLoading(false);
      return;
    }

    try {
      const response = await checkUsageStatus(token);
      if (response.success && response.data) {
        // Parse usage info from the response
        // The backend returns hasReachedLimit as boolean, but we need counts
        // We'll get this from ocrResult.trialInfo if available
        setUsageData({
          hasReachedLimit: response.data
        });
      }
    } catch (error) {
      console.error('Error fetching usage data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsageData();
  }, [token, isAuthenticated]);

  // Update when OCR result changes (new scan)
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

  // Don't show usage counter for admins
  if (isAdmin()) {
    return null;
  }

  if (!isAuthenticated() || loading) {
    return null;
  }

  if (!usageData) {
    return null;
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
            <button className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white px-5 py-2.5 rounded-lg font-semibold hover:from-yellow-600 hover:to-orange-600 transition-all shadow-md hover:shadow-lg flex items-center gap-2">
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

