import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const AROverlay = ({ image, characters, showTranslation = false, translations = {} }) => {
  const [imageUrl, setImageUrl] = useState(null)
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 })
  const [containerDimensions, setContainerDimensions] = useState({ width: 0, height: 0 })
  const [hoveredChar, setHoveredChar] = useState(null)
  const [scale, setScale] = useState(1)
  const imgRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    if (image instanceof File) {
      const reader = new FileReader()
      reader.onload = (e) => setImageUrl(e.target.result)
      reader.readAsDataURL(image)
    } else {
      setImageUrl(image)
    }
  }, [image])

  useEffect(() => {
    const updateDimensions = () => {
      if (imgRef.current && containerRef.current) {
        const img = imgRef.current
        const container = containerRef.current
        
        // Get natural image dimensions
        const naturalWidth = img.naturalWidth || img.width
        const naturalHeight = img.naturalHeight || img.height
        
        // Get displayed dimensions
        const displayedWidth = img.clientWidth
        const displayedHeight = img.clientHeight
        
        // Calculate scale factor
        const scaleX = displayedWidth / naturalWidth
        const scaleY = displayedHeight / naturalHeight
        
        setImageDimensions({ width: naturalWidth, height: naturalHeight })
        setContainerDimensions({ width: displayedWidth, height: displayedHeight })
        setScale(Math.min(scaleX, scaleY))
      }
    }

    if (imageUrl) {
      const img = imgRef.current
      if (img) {
        if (img.complete) {
          updateDimensions()
        } else {
          img.onload = updateDimensions
        }
      }
    }

    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [imageUrl])

  if (!imageUrl) {
    return (
      <div className="card text-center py-12">
        <p className="text-gray-500">No image loaded</p>
      </div>
    )
  }

  const getScaledBbox = (bbox) => {
    if (!bbox || !imageDimensions.width) return null
    
    return {
      x: bbox.x * scale,
      y: bbox.y * scale,
      width: bbox.width * scale,
      height: bbox.height * scale
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'bg-green-500'
    if (confidence >= 0.7) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  return (
    <div className="card bg-white">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold flex items-center text-primary-600">
          <span className="mr-3 text-4xl">👓</span>
          AR Visualization
        </h2>
        {characters && characters.length > 0 && (
          <span className="text-sm font-semibold text-gray-600 bg-blue-100 px-3 py-1 rounded-full">
            {characters.length} characters detected
          </span>
        )}
      </div>
      
      <div 
        ref={containerRef}
        className="relative w-full bg-gray-900 rounded-xl overflow-hidden shadow-2xl"
        style={{ minHeight: '400px' }}
      >
        <img
          ref={imgRef}
          src={imageUrl}
          alt="AR Overlay"
          className="w-full h-auto max-h-[70vh] object-contain mx-auto"
          style={{ display: 'block' }}
        />
        
        {characters && characters.map((char, idx) => {
          const scaledBbox = getScaledBbox(char.bbox)
          if (!scaledBbox) return null
          
          const isHovered = hoveredChar === idx
          const confidence = char.confidence || 0
          const translation = translations[char.character] || ''
          
          return (
            <motion.div
              key={idx}
              className="absolute cursor-pointer group"
              style={{
                left: `${scaledBbox.x}px`,
                top: `${scaledBbox.y}px`,
                width: `${Math.max(scaledBbox.width, 20)}px`,
                height: `${Math.max(scaledBbox.height, 20)}px`,
              }}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ 
                opacity: 1, 
                scale: 1,
                borderColor: isHovered ? 'var(--primary-color)' : 'var(--primary-color)'
              }}
              transition={{ delay: idx * 0.05, duration: 0.3 }}
              onMouseEnter={() => setHoveredChar(idx)}
              onMouseLeave={() => setHoveredChar(null)}
            >
              {/* Bounding Box */}
              <div
                className={`absolute inset-0 rounded transition-all duration-300 ${
                  isHovered 
                    ? 'border-3 border-primary-600 bg-primary-600 bg-opacity-30 shadow-lg shadow-primary-600/50' 
                    : 'border-2 border-primary-400 bg-primary-400 bg-opacity-20'
                }`}
                style={{
                  boxShadow: isHovered ? '0 0 20px rgba(41, 82, 255, 0.6)' : 'none'
                }}
              />
              
              {/* Confidence Indicator */}
              <div 
                className={`absolute -top-2 -right-2 w-4 h-4 rounded-full ${getConfidenceColor(confidence)} border-2 border-white shadow-lg`}
                title={`Confidence: ${(confidence * 100).toFixed(1)}%`}
              />
              
              {/* Character Label (shown in box if space permits) */}
              {scaledBbox.width > 30 && scaledBbox.height > 30 && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-white font-bold text-lg drop-shadow-lg" style={{ 
                    fontSize: `${Math.min(scaledBbox.width * 0.4, scaledBbox.height * 0.5)}px`,
                    textShadow: '0 2px 4px rgba(0,0,0,0.8)'
                  }}>
                    {char.character}
                  </span>
                </div>
              )}
            </motion.div>
          )
        })}
        
        {/* Hover Tooltip */}
        <AnimatePresence>
          {hoveredChar !== null && characters[hoveredChar] && (
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.9 }}
              className="absolute z-50 bg-gray-900 text-white px-4 py-3 rounded-lg shadow-2xl border border-gray-700"
              style={{
                left: `${getScaledBbox(characters[hoveredChar].bbox)?.x || 0}px`,
                top: `${(getScaledBbox(characters[hoveredChar].bbox)?.y || 0) - 60}px`,
                pointerEvents: 'none'
              }}
            >
              <div className="flex items-center space-x-3">
                <div className="text-2xl font-bold">{characters[hoveredChar].character}</div>
                <div className="h-6 w-px bg-gray-600"></div>
                <div>
                  <div className="text-xs text-gray-400">Confidence</div>
                  <div className="text-sm font-semibold">
                    {(characters[hoveredChar].confidence * 100 || 0).toFixed(1)}%
                  </div>
                </div>
                {showTranslation && translations[characters[hoveredChar].character] && (
                  <>
                    <div className="h-6 w-px bg-gray-600"></div>
                    <div>
                      <div className="text-xs text-gray-400">Translation</div>
                      <div className="text-sm font-semibold">
                        {translations[characters[hoveredChar].character]}
                      </div>
                    </div>
                  </>
                )}
              </div>
              {/* Arrow */}
              <div className="absolute bottom-0 left-6 transform translate-y-full">
                <div className="w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-gray-900"></div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      <div className="mt-4 flex items-center justify-between text-sm text-gray-600">
        <p className="flex items-center">
          <span className="mr-2">💡</span>
          Hover over highlighted boxes to see character details
        </p>
        {characters && characters.length > 0 && (
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span>High confidence (≥90%)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span>Medium (70-89%)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Low (&lt;70%)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default AROverlay

