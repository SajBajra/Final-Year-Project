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
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6 space-y-4 md:space-y-0">
        <div className="flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-primary-600">
            <FaSearch className="text-4xl text-white" />
          </div>
          <div>
            <h2 className="text-3xl font-black text-primary-600">
              Recognition Result
            </h2>
            {confidence !== undefined && (
              <div className="flex items-center space-x-2 mt-1">
                <div className="flex items-center">
                  <span className="text-xs text-gray-500 mr-2">Confidence:</span>
                  <div className="flex items-center space-x-1">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <span
                        key={star}
                        className={`text-sm ${
                          star <= Math.round(confidence * 5)
                            ? 'text-yellow-400'
                            : 'text-gray-300'
                        }`}
                      >
                        ★
                      </span>
                    ))}
                  </div>
                  <span className="text-xs font-semibold text-gray-600 ml-2">
                    {(confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="flex flex-wrap items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onToggleAR}
            className={`px-5 py-2.5 rounded-lg font-semibold transition-all duration-200 shadow-md ${
              showAR 
                ? 'bg-primary-600 text-white hover:bg-primary-700 hover:shadow-lg' 
                : 'bg-white text-secondary-500 border-2 border-gray-300 hover:border-primary-600 hover:bg-gray-50'
            }`}
          >
            <span className="flex items-center gap-2">
              <FaEye className="inline" /> {showAR ? 'Hide AR' : 'Show AR Overlay'}
            </span>
          </motion.button>
          
          {text && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setShowTranslation(!showTranslation)}
              disabled={translationLoading}
              className="px-5 py-2.5 rounded-lg font-semibold bg-secondary-500 text-white hover:bg-secondary-600 hover:shadow-lg disabled:opacity-50 transition-all duration-200 shadow-md"
            >
              {translationLoading ? (
                <span className="flex items-center space-x-2">
                  <FaSpinner className="animate-spin" />
                  <span>Translating...</span>
                </span>
              ) : showTranslation ? (
                <span className="flex items-center gap-2">
                  <FaGlobe /> Hide Translation
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <FaGlobe /> Show Translation
                </span>
              )}
            </motion.button>
          )}
        </div>
      </div>
      
      {/* Text Display */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`rounded-xl p-8 mb-6 border-2 shadow-sm ${
          (text && (text.includes('Error') || text.includes('error')))
            ? 'bg-red-50 border-red-200'
            : 'bg-gray-50 border-gray-200'
        }`}
      >
        <div className="mb-4">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">
            {text && (text.includes('Error') || text.includes('error')) ? 'Error Message' : 'Recognized Text (Devanagari)'}
          </h3>
          <p className={`text-3xl md:text-4xl font-bold leading-relaxed break-words ${
            (text && (text.includes('Error') || text.includes('error')))
              ? 'text-red-600'
              : 'text-gray-800'
          }`} style={{ fontFamily: 'Arial, sans-serif', unicodeBidi: 'embed' }}>
            {text || 'No text detected'}
          </p>
        </div>
        
        {characters && characters.length > 0 && (
          <div className="mt-6 pt-6 border-t border-gray-200/50">
            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4">
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
                  <div className={`px-3 py-2 rounded-lg font-semibold text-sm shadow-md transition-all duration-300 ${
                    char.confidence >= 0.8
                      ? 'bg-green-100 text-green-800 border border-green-300'
                      : char.confidence >= 0.6
                      ? 'bg-yellow-100 text-yellow-800 border border-yellow-300'
                      : 'bg-red-100 text-red-800 border border-red-300'
                  }`}>
                    {char.character}
                    {showTranslation && translations[char.character] && (
                      <span className="ml-2 text-xs opacity-75">
                        ({translations[char.character]})
                      </span>
                    )}
                  </div>
                  
                  {/* Confidence Tooltip */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none whitespace-nowrap">
                    Confidence: {(char.confidence * 100).toFixed(1)}%
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
      
      {/* Translation Section - Only show English translation option */}
      {showTranslation && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-gray-50 rounded-xl p-6 border-2 border-gray-200 space-y-4"
        >
          <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
            <FaGlobe className="mr-2" />
            English Translation (Optional)
          </h3>
          
          {/* Devanagari Text Display (for context in translation section) */}
          {devanagariText && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg p-4 border border-gray-300"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-bold text-gray-600 uppercase tracking-wider">
                  Devanagari (देवनागरी) - Original Output
                </span>
              </div>
              <p className="text-2xl font-bold text-gray-800 leading-relaxed">
                {devanagariText}
              </p>
            </motion.div>
          )}
          
          {/* English Translation */}
          {englishText ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-lg p-4 border border-gray-300"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-bold text-gray-600 uppercase tracking-wider">
                  English
                </span>
              </div>
              <p className="text-xl font-semibold text-gray-800 leading-relaxed">
                {englishText}
              </p>
            </motion.div>
          ) : (
            <motion.button
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              onClick={onTranslateToEnglish}
              disabled={translationLoading || !devanagariText}
              className="w-full bg-primary-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-primary-700 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {translationLoading ? (
                <span className="flex items-center justify-center space-x-2">
                  <FaSpinner className="animate-spin" />
                  <span>Translating to English...</span>
                </span>
              ) : (
                <span className="flex items-center justify-center space-x-2">
                  <FaFlag />
                  <span>Translate to English</span>
                </span>
              )}
            </motion.button>
          )}
        </motion.div>
      )}
    </motion.div>
  )
}

export default OCRResult

