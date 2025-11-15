import { motion } from 'framer-motion'
import { FaSearch, FaEye, FaGlobe, FaSpinner, FaFlag } from 'react-icons/fa'

const OCRResult = ({ 
  text, 
  characters, 
  confidence, 
  onToggleAR, 
  showAR,
  translations = {},
  devanagariText = '',
  englishText = '',
  showTranslation = false,
  setShowTranslation,
  translationLoading = false,
  onTranslateToEnglish
}) => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="card"
    >
      {/* Header Section - Responsive */}
      <div className="flex flex-col space-y-4 sm:space-y-0 sm:flex-row sm:items-center sm:justify-between mb-4 sm:mb-6">
        <div className="flex items-center space-x-3 sm:space-x-4">
          <div className="p-2 sm:p-3 rounded-xl bg-primary-600 flex-shrink-0">
            <FaSearch className="text-2xl sm:text-3xl md:text-4xl text-white" />
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-xl sm:text-2xl md:text-3xl font-black text-primary-600 truncate">
              Recognition Result
            </h2>
            {confidence !== undefined && (
              <div className="flex flex-wrap items-center gap-2 sm:gap-3 mt-1 sm:mt-2">
                <span className="text-xs text-gray-500">Confidence:</span>
                <div className="flex items-center space-x-1">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <span
                      key={star}
                      className={`text-xs sm:text-sm ${
                        star <= Math.round(confidence * 5)
                          ? 'text-yellow-400'
                          : 'text-gray-300'
                      }`}
                    >
                      ★
                    </span>
                  ))}
                </div>
                <span className="text-xs sm:text-sm font-semibold text-gray-600">
                  {(confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        </div>
        
        {/* Action Buttons - Responsive */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onToggleAR}
            className={`px-4 sm:px-5 py-2 sm:py-2.5 rounded-lg font-semibold text-sm sm:text-base transition-all duration-200 shadow-md flex items-center justify-center gap-2 ${
              showAR 
                ? 'bg-primary-600 text-white hover:bg-primary-700 hover:shadow-lg' 
                : 'bg-white text-secondary-500 border-2 border-gray-300 hover:border-primary-600 hover:bg-gray-50'
            }`}
          >
            <FaEye className="text-base sm:text-lg" />
            <span className="hidden sm:inline">{showAR ? 'Hide AR' : 'Show AR Overlay'}</span>
            <span className="sm:hidden">{showAR ? 'Hide' : 'AR'}</span>
          </motion.button>
          
          {text && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setShowTranslation(!showTranslation)}
              disabled={translationLoading}
              className="px-4 sm:px-5 py-2 sm:py-2.5 rounded-lg font-semibold text-sm sm:text-base bg-secondary-500 text-white hover:bg-secondary-600 hover:shadow-lg disabled:opacity-50 transition-all duration-200 shadow-md flex items-center justify-center gap-2"
            >
              {translationLoading ? (
                <>
                  <FaSpinner className="animate-spin text-base sm:text-lg" />
                  <span>Translating...</span>
                </>
              ) : showTranslation ? (
                <>
                  <FaGlobe className="text-base sm:text-lg" />
                  <span className="hidden sm:inline">Hide Translation</span>
                  <span className="sm:hidden">Hide</span>
                </>
              ) : (
                <>
                  <FaGlobe className="text-base sm:text-lg" />
                  <span className="hidden sm:inline">Show Translation</span>
                  <span className="sm:hidden">Translate</span>
                </>
              )}
            </motion.button>
          )}
        </div>
      </div>
      
      {/* Text Display - Responsive */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`rounded-xl p-4 sm:p-6 md:p-8 mb-4 sm:mb-6 border-2 shadow-sm ${
          (text && (text.includes('Error') || text.includes('error')))
            ? 'bg-red-50 border-red-200'
            : 'bg-gray-50 border-gray-200'
        }`}
      >
        <div className="mb-3 sm:mb-4">
          <h3 className="text-xs sm:text-sm font-bold text-gray-500 uppercase tracking-wider mb-2 sm:mb-3">
            {text && (text.includes('Error') || text.includes('error')) ? 'Error Message' : 'Recognized Text (Devanagari)'}
          </h3>
          <p className={`text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold leading-relaxed break-words ${
            (text && (text.includes('Error') || text.includes('error')))
              ? 'text-red-600'
              : 'text-gray-800'
          }`} style={{ fontFamily: 'Arial, sans-serif', unicodeBidi: 'embed' }}>
            {text || 'No text detected'}
          </p>
        </div>
        
        {/* Character Breakdown - Responsive */}
        {characters && characters.length > 0 && (
          <div className="mt-4 sm:mt-6 pt-4 sm:pt-6 border-t border-gray-200/50">
            <h3 className="text-xs sm:text-sm font-bold text-gray-500 uppercase tracking-wider mb-3 sm:mb-4">
              Character Breakdown ({characters.length} characters)
            </h3>
            <div className="flex flex-wrap gap-2">
              {characters.map((char, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: idx * 0.02 }}
                  whileHover={{ scale: 1.1, y: -2 }}
                  className="relative group"
                >
                  <div className={`px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg font-semibold text-xs sm:text-sm shadow-md transition-all duration-300 ${
                    char.confidence >= 0.8
                      ? 'bg-green-100 text-green-800 border border-green-300'
                      : char.confidence >= 0.6
                      ? 'bg-yellow-100 text-yellow-800 border border-yellow-300'
                      : 'bg-red-100 text-red-800 border border-red-300'
                  }`}>
                    {char.character}
                    {showTranslation && translations[char.character] && (
                      <span className="ml-1 sm:ml-2 text-xs opacity-75 hidden sm:inline">
                        ({translations[char.character]})
                      </span>
                    )}
                  </div>
                  
                  {/* Confidence Tooltip */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap z-10">
                    Confidence: {(char.confidence * 100).toFixed(1)}%
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
      
      {/* Translation Section - Responsive */}
      {showTranslation && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-gray-50 rounded-xl p-4 sm:p-6 border-2 border-gray-200 space-y-3 sm:space-y-4"
        >
          <h3 className="text-base sm:text-lg font-bold text-gray-800 mb-3 sm:mb-4 flex items-center">
            <FaGlobe className="mr-2 text-primary-600" />
            English Translation (Optional)
          </h3>
          
          {/* Devanagari Text Display */}
          {devanagariText && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg p-3 sm:p-4 border border-gray-300"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs sm:text-sm font-bold text-gray-600 uppercase tracking-wider">
                  Devanagari (देवनागरी) - Original
                </span>
              </div>
              <p className="text-lg sm:text-xl md:text-2xl font-bold text-gray-800 leading-relaxed break-words" style={{ fontFamily: 'Arial, sans-serif', unicodeBidi: 'embed' }}>
                {devanagariText}
              </p>
            </motion.div>
          )}
          
          {/* English Translation */}
          {englishText ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg p-3 sm:p-4 border border-gray-300"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs sm:text-sm font-bold text-gray-600 uppercase tracking-wider">
                  English
                </span>
              </div>
              <p className="text-base sm:text-lg md:text-xl font-semibold text-gray-800 leading-relaxed break-words">
                {englishText}
              </p>
            </motion.div>
          ) : (
            <motion.button
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              onClick={onTranslateToEnglish}
              disabled={translationLoading || !devanagariText}
              className="w-full bg-primary-600 text-white font-semibold py-2.5 sm:py-3 px-4 sm:px-6 rounded-lg hover:bg-primary-700 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center space-x-2 text-sm sm:text-base"
            >
              {translationLoading ? (
                <>
                  <FaSpinner className="animate-spin text-base sm:text-lg" />
                  <span>Translating to English...</span>
                </>
              ) : (
                <>
                  <FaFlag className="text-base sm:text-lg" />
                  <span>Translate to English</span>
                </>
              )}
            </motion.button>
          )}
        </motion.div>
      )}
    </motion.div>
  )
}

export default OCRResult
