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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card bg-white shadow-lg"
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold flex items-center bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          <span className="mr-3 text-4xl">🔍</span>
          Recognition Result
        </h2>
        <div className="flex items-center space-x-3">
          <button
            onClick={onToggleAR}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              showAR 
                ? 'bg-blue-600 text-white hover:bg-blue-700' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {showAR ? '👓 Hide AR' : '👓 Show AR Overlay'}
          </button>
          {text && (
            <button
              onClick={() => setShowTranslation(!showTranslation)}
              disabled={translationLoading}
              className="px-4 py-2 rounded-lg font-semibold bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 transition-all"
            >
              {translationLoading ? '⏳ Translating...' : showTranslation ? '🌐 Hide Translation' : '🌐 Show Translation'}
            </button>
          )}
        </div>
      </div>
      
      <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-8 mb-6">
        <div className="mb-3">
          <span className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Recognized Text:</span>
        </div>
        <div className="text-5xl font-bold text-gray-900 mb-6 min-h-[4rem] flex items-center">
          {text || 'No text detected'}
        </div>
        
        {showTranslation && translations._full && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 pt-4 border-t border-gray-300"
          >
            <span className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Translation:</span>
            <div className="text-2xl font-semibold text-gray-800 mt-2">
              {translations._full}
            </div>
          </motion.div>
        )}
        
        {confidence > 0 && (
          <div className="flex items-center space-x-3 mt-4">
            <span className="text-sm font-semibold text-gray-600">Confidence:</span>
            <div className="flex-1 bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all duration-500 ${
                  confidence >= 90 ? 'bg-green-500' : 
                  confidence >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${Math.min(confidence, 100)}%` }}
              ></div>
            </div>
            <span className="text-sm font-bold text-gray-700 min-w-[4rem]">{confidence.toFixed(1)}%</span>
          </div>
        )}
      </div>
      
      {characters && characters.length > 0 && (
        <div>
          <h3 className="font-bold text-xl mb-4 text-gray-800">Character Breakdown:</h3>
          <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-3">
            {characters.map((char, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: idx * 0.05 }}
                className="bg-gray-50 border-2 border-gray-200 rounded-lg p-4 text-center hover:bg-blue-50 hover:border-blue-400 transition-all transform hover:scale-105"
              >
                <div className="text-3xl font-bold mb-2">{char.character}</div>
                {showTranslation && translations[char.character] && (
                  <div className="text-xs text-green-600 font-semibold mb-1">
                    {translations[char.character]}
                  </div>
                )}
                <div className={`text-xs font-semibold ${
                  (char.confidence * 100) >= 90 ? 'text-green-600' :
                  (char.confidence * 100) >= 70 ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {char.confidence ? `${(char.confidence * 100).toFixed(0)}%` : '-'}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}

export default OCRResult

