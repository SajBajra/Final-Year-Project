import { motion } from 'framer-motion';
import { FaExclamationTriangle, FaCheckCircle } from 'react-icons/fa';
import { Link } from 'react-router-dom';

const TrialCounter = ({ trialInfo, onLoginClick }) => {
  if (!trialInfo) return null;

  const { remainingTrials, usedTrials, maxTrials, requiresLogin } = trialInfo;

  if (requiresLogin) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-4 p-4 bg-secondary-50 border-2 border-secondary-300 rounded-lg"
      >
        <div className="flex items-start gap-3">
          <FaExclamationTriangle className="text-orange-600 text-xl mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <h3 className="font-bold text-orange-900 mb-1">Trial Limit Reached</h3>
            <p className="text-sm text-orange-800 mb-3">
              You've used all {maxTrials} free OCR attempts. Create an account or login to continue using Lipika OCR.
            </p>
            <div className="flex gap-2">
              <Link
                to="/register"
                className="btn-primary text-sm px-4 py-2"
              >
                Create Account
              </Link>
              <Link
                to="/login"
                className="btn-outline text-sm px-4 py-2"
              >
                Login
              </Link>
            </div>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FaCheckCircle className="text-blue-600" />
          <span className="text-sm font-medium text-blue-900">
            Free Trials Remaining: <span className="font-bold">{remainingTrials}/{maxTrials}</span>
          </span>
        </div>
        <div className="text-xs text-blue-700">
          {usedTrials} used
        </div>
      </div>
      <div className="mt-2 w-full bg-blue-200 rounded-full h-2">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${(remainingTrials / maxTrials) * 100}%` }}
          transition={{ duration: 0.5 }}
          className="bg-blue-600 h-2 rounded-full"
        />
      </div>
      <p className="text-xs text-blue-700 mt-2">
        Create an account for unlimited OCR access
      </p>
    </motion.div>
  );
};

export default TrialCounter;

