import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const AROverlay = ({ image, characters, showTranslation = false, translations = {} }) => {
  const [imageUrl, setImageUrl] = useState(null)
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 })
  const [containerDimensions, setContainerDimensions] = useState({ width: 0, height: 0 })
  const [hoveredChar, setHoveredChar] = useState(null)
  const [scale, setScale] = useState(1)
  const [isMobile, setIsMobile] = useState(false)
  const imgRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    // Detect mobile device
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

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
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4 sm:mb-6">
        <h2 className="text-xl sm:text-2xl md:text-3xl font-bold flex items-center text-primary-600">
          <span className="mr-2 sm:mr-3 text-2xl sm:text-3xl md:text-4xl">👓</span>
          <span className="whitespace-nowrap">AR Visualization</span>
        </h2>
        {characters && characters.length > 0 && (
          <span className="text-xs sm:text-sm font-semibold text-gray-600 bg-blue-100 px-2 sm:px-3 py-1 rounded-full w-fit">
            {characters.length} character{characters.length !== 1 ? 's' : ''} detected
          </span>
        )}
      </div>
      
      <div 
        ref={containerRef}
        className="relative w-full bg-gray-900 rounded-xl overflow-hidden shadow-2xl"
        style={{ minHeight: isMobile ? '300px' : '400px' }}
        onClick={(e) => {
          if (isMobile && e.target === e.currentTarget) {
            setHoveredChar(null)
          }
        }}
      >
        <img
          ref={imgRef}
          src={imageUrl}
          alt="AR Overlay"
          className="w-full h-auto max-h-[60vh] sm:max-h-[70vh] object-contain mx-auto"
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
              className="absolute cursor-pointer group touch-manipulation"
              style={{
                left: `${scaledBbox.x}px`,
                top: `${scaledBbox.y}px`,
                width: `${Math.max(scaledBbox.width, isMobile ? 30 : 20)}px`,
                height: `${Math.max(scaledBbox.height, isMobile ? 30 : 20)}px`,
              }}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ 
                opacity: 1, 
                scale: 1,
                borderColor: isHovered ? 'var(--primary-color)' : 'var(--primary-color)'
              }}
              transition={{ delay: idx * 0.05, duration: 0.3 }}
              onMouseEnter={() => !isMobile && setHoveredChar(idx)}
              onMouseLeave={() => !isMobile && setHoveredChar(null)}
              onClick={() => {
                if (isMobile) {
                  setHoveredChar(hoveredChar === idx ? null : idx)
                }
              }}
              onTouchStart={(e) => {
                e.stopPropagation()
                setHoveredChar(hoveredChar === idx ? null : idx)
              }}
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
        
        {/* Hover/Tap Tooltip */}
        <AnimatePresence>
          {hoveredChar !== null && characters[hoveredChar] && (() => {
            const bbox = getScaledBbox(characters[hoveredChar].bbox)
            const tooltipX = Math.max(10, Math.min((bbox?.x || 0), containerDimensions.width - (isMobile ? 200 : 300)))
            const tooltipY = Math.max(10, (bbox?.y || 0) - (isMobile ? 80 : 60))
            
            return (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.9 }}
                className="absolute z-50 bg-gray-900 text-white rounded-lg shadow-2xl border border-gray-700"
                style={{
                  left: `${tooltipX}px`,
                  top: `${tooltipY}px`,
                  pointerEvents: isMobile ? 'auto' : 'none',
                  maxWidth: isMobile ? '200px' : '400px',
                  padding: isMobile ? '8px 12px' : '12px 16px'
                }}
                onClick={(e) => isMobile && e.stopPropagation()}
              >
                <div className={`flex ${isMobile ? 'flex-col gap-2' : 'items-center space-x-3'}`}>
                  <div className={`${isMobile ? 'text-xl' : 'text-2xl'} font-bold`}>
                    {characters[hoveredChar].character}
                  </div>
                  {!isMobile && <div className="h-6 w-px bg-gray-600"></div>}
                  <div>
                    <div className="text-xs text-gray-400">Confidence</div>
                    <div className={`${isMobile ? 'text-xs' : 'text-sm'} font-semibold`}>
                      {(characters[hoveredChar].confidence * 100 || 0).toFixed(1)}%
                    </div>
                  </div>
                  {showTranslation && translations[characters[hoveredChar].character] && (
                    <>
                      {!isMobile && <div className="h-6 w-px bg-gray-600"></div>}
                      <div>
                        <div className="text-xs text-gray-400">Translation</div>
                        <div className={`${isMobile ? 'text-xs' : 'text-sm'} font-semibold`}>
                          {translations[characters[hoveredChar].character]}
                        </div>
                      </div>
                    </>
                  )}
                </div>
                {/* Arrow */}
                {!isMobile && (
                  <div className="absolute bottom-0 left-6 transform translate-y-full">
                    <div className="w-0 h-0 border-l-8 border-r-8 border-t-8 border-transparent border-t-gray-900"></div>
                  </div>
                )}
              </motion.div>
            )
          })()}
        </AnimatePresence>
      </div>
      
      <div className="mt-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 text-xs sm:text-sm text-gray-600">
        <p className="flex items-center">
          <span className="mr-2">💡</span>
          <span>{isMobile ? 'Tap' : 'Hover over'} highlighted boxes to see character details</span>
        </p>
        {characters && characters.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 sm:gap-4">
            <div className="flex items-center space-x-1 sm:space-x-2">
              <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full bg-green-500"></div>
              <span className="text-xs sm:text-sm whitespace-nowrap">High (≥90%)</span>
            </div>
            <div className="flex items-center space-x-1 sm:space-x-2">
              <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full bg-yellow-500"></div>
              <span className="text-xs sm:text-sm whitespace-nowrap">Medium (70-89%)</span>
            </div>
            <div className="flex items-center space-x-1 sm:space-x-2">
              <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full bg-red-500"></div>
              <span className="text-xs sm:text-sm whitespace-nowrap">Low (&lt;70%)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default AROverlay

