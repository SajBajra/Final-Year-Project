import { motion } from 'framer-motion'

const OCRResult = ({ 
  text, 
  characters, 
  confidence, 
  onToggleAR, 
  showAR,
  translations = {},
  showTranslation = false,
  setShowTranslation,
  translationLoading = false,
  onTranslate
}) => {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="card-glow"
    >
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6 space-y-4 md:space-y-0">
        <div className="flex items-center space-x-4">
          <div className="p-3 rounded-xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500">
            <span className="text-4xl">🔍</span>
          </div>
          <div>
            <h2 className="text-3xl font-black text-gradient">
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
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onToggleAR}
            className={`px-5 py-2.5 rounded-xl font-semibold transition-all duration-300 shadow-lg ${
              showAR 
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-xl' 
                : 'bg-white/80 backdrop-blur-sm text-gray-700 border-2 border-gray-300/50 hover:border-purple-400 hover:bg-white'
            }`}
          >
            {showAR ? '👓 Hide AR' : '👓 Show AR Overlay'}
          </motion.button>
          
          {text && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowTranslation(!showTranslation)}
              disabled={translationLoading}
              className="px-5 py-2.5 rounded-xl font-semibold bg-gradient-to-r from-green-500 to-emerald-600 text-white hover:shadow-xl disabled:opacity-50 transition-all duration-300 shadow-lg"
            >
              {translationLoading ? (
                <span className="flex items-center space-x-2">
                  <span className="animate-spin">⏳</span>
                  <span>Translating...</span>
                </span>
              ) : showTranslation ? (
                '🌐 Hide Translation'
              ) : (
                '🌐 Show Translation'
              )}
            </motion.button>
          )}
        </div>
      </div>
      
      {/* Text Display */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`rounded-2xl p-8 mb-6 border-2 shadow-inner ${
          (text && (text.includes('Error') || text.includes('error')))
            ? 'bg-gradient-to-br from-red-50 via-orange-50 to-pink-50 border-red-200'
            : 'bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 border-white/50'
        }`}
      >
        <div className="mb-4">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">
            {text && (text.includes('Error') || text.includes('error')) ? 'Error Message' : 'Recognized Text'}
          </h3>
          <p className={`text-3xl md:text-4xl font-bold leading-relaxed break-words ${
            (text && (text.includes('Error') || text.includes('error')))
              ? 'text-red-600'
              : 'text-gray-800'
          }`}>
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
      
      {/* Translation Section */}
      {showTranslation && Object.keys(translations).length > 0 && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-6 border-2 border-green-200/50"
        >
          <h3 className="text-lg font-bold text-gray-800 mb-4 flex items-center">
            <span className="mr-2">🌐</span>
            Translations
          </h3>
          <div className="grid md:grid-cols-2 gap-3">
            {Object.entries(translations).map(([char, translation], idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.05 }}
                className="bg-white/60 backdrop-blur-sm rounded-lg p-3 border border-white/50"
              >
                <span className="font-bold text-gray-700">{char}</span>
                <span className="mx-2 text-gray-400">→</span>
                <span className="text-gray-600">{translation}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

export default OCRResult

