const OCRResult = ({ text, characters, confidence, onToggleAR, showAR }) => {
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold flex items-center">
          <span className="mr-3">🔍</span>
          Recognition Result
        </h2>
        <button
          onClick={onToggleAR}
          className={`btn-${showAR ? 'primary' : 'secondary'}`}
        >
          {showAR ? '👓 Hide AR' : '👓 Show AR Overlay'}
        </button>
      </div>
      
      <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6 mb-4">
        <div className="mb-2">
          <span className="text-sm font-semibold text-gray-600">Recognized Text:</span>
        </div>
        <div className="text-4xl font-bold text-gray-900 mb-4 min-h-[3rem] flex items-center">
          {text || 'No text detected'}
        </div>
        {confidence > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Confidence:</span>
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${confidence}%` }}
              ></div>
            </div>
            <span className="text-sm font-semibold text-gray-700">{confidence.toFixed(1)}%</span>
          </div>
        )}
      </div>
      
      {characters && characters.length > 0 && (
        <div>
          <h3 className="font-semibold text-lg mb-3">Character Breakdown:</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {characters.map((char, idx) => (
              <div
                key={idx}
                className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-center hover:bg-blue-50 hover:border-blue-300 transition-colors"
              >
                <div className="text-3xl font-bold mb-1">{char.character}</div>
                <div className="text-xs text-gray-500">
                  {char.confidence ? `${char.confidence.toFixed(1)}%` : '-'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default OCRResult

