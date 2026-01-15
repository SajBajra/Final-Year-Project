import { useState, useEffect, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { FaScroll, FaCamera, FaSearch, FaEye, FaUpload, FaArrowRight, FaImage } from 'react-icons/fa'
import Webcam from 'react-webcam'
import heroImage from '../images/HeroSection-FrontPage.png'
import { useAuth } from '../context/AuthContext'
import { getOrCreateTrialCookie } from '../utils/cookieUtils'
import ImageUpload from '../components/ImageUpload'
import CameraCapture from '../components/CameraCapture'
import OCRResult from '../components/OCRResult'
import AROverlay from '../components/AROverlay'
import TrialCounter from '../components/TrialCounter'
import UserUsageCounter from '../components/UserUsageCounter'
import { recognizeText, translateText } from '../services/ocrService'

function Home() {
  const [image, setImage] = useState(null)
  const [ocrResult, setOcrResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showAR, setShowAR] = useState(false)
  const [translations, setTranslations] = useState({})
  const [showTranslation, setShowTranslation] = useState(false)
  const [translationLoading, setTranslationLoading] = useState(false)
  const [translationError, setTranslationError] = useState(null)
  const [devanagariText, setDevanagariText] = useState('')
  const [englishText, setEnglishText] = useState('')
  const [trialInfo, setTrialInfo] = useState(null)
  const [isMobile, setIsMobile] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [preview, setPreview] = useState(null)
  
  const webcamRef = useRef(null)
  const fileInputRef = useRef(null)
  const { isAuthenticated, getAuthHeaders, isAdmin } = useAuth()
  const cookieId = getOrCreateTrialCookie()
  
  useEffect(() => {
    // Set cookie if not already set
    if (!document.cookie.includes('lipika_trial_id')) {
      getOrCreateTrialCookie()
    }
    
    // Detect mobile view
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }
    
    checkMobile()
    window.addEventListener('resize', checkMobile)
    
    return () => window.removeEventListener('resize', checkMobile)
  }, [])
  
  // Auto-start camera on mobile
  useEffect(() => {
    if (isMobile && !ocrResult && !loading && !image) {
      setCameraActive(true)
    }
  }, [isMobile, ocrResult, loading, image])

  const handleImageUpload = (file) => {
    setImage(file)
    setOcrResult(null)
    setShowAR(false)
    setTranslations({})
    setDevanagariText('')
    setEnglishText('')
    setShowTranslation(false)
  }

  const handleOCRComplete = async (result) => {
    setOcrResult(result)
    setLoading(false)
    
    // Update trial info if present
    if (result?.trialInfo) {
      setTrialInfo(result.trialInfo)
    }
    
    // Check if trial limit exceeded (but admins bypass this)
    if (result?.error && result?.trialInfo?.requiresLogin && !isAdmin()) {
      // Don't show OCR result, just show trial limit message
      return
    }
    
    // OCR service returns Devanagari text directly from the model
    // Ensure we preserve the Devanagari characters exactly as returned
    if (result && result.text) {
      // Set Devanagari text directly - no translation needed
      setDevanagariText(result.text)
      setTranslations({ devanagari: result.text })
      console.log('OCR Result - Devanagari text:', result.text)
      console.log('OCR Result - Characters:', result.characters)
      
      // Show AR overlay by default when OCR completes successfully
      if (result.characters && result.characters.length > 0) {
        setShowAR(true)
      }
    }
  }

  const handleProcessing = () => {
    setLoading(true)
  }

  const toggleAR = () => {
    setShowAR(!showAR)
  }

  const handleTranslateToEnglish = useCallback(async () => {
    setTranslationLoading(true)
    setTranslationError(null)
    
    try {
      // Translate Devanagari to English via API
      const textToTranslate = devanagariText || ocrResult?.text || ''
      
      if (!textToTranslate || textToTranslate.trim() === '') {
        setTranslationError('No text available to translate')
        return
      }
      
      const englishResult = await translateText(textToTranslate, 'en')
      if (englishResult && englishResult.trim() !== '') {
        setEnglishText(englishResult)
        setTranslations({ ...translations, english: englishResult })
        setTranslationError(null) // Clear any previous errors
        // Replace the displayed text with translated text
        if (ocrResult) {
          setOcrResult({
            ...ocrResult,
            text: englishResult
          })
        }
      } else {
        setTranslationError('Translation returned empty result')
      }
    } catch (error) {
      console.error('English translation error:', error)
      setTranslationError(error.message || 'Translation failed. Please try again.')
    } finally {
      setTranslationLoading(false)
    }
  }, [devanagariText, ocrResult, translations])

  // Auto-translate when showTranslation is toggled to true
  useEffect(() => {
    if (showTranslation && !englishText && devanagariText && !translationLoading) {
      handleTranslateToEnglish()
    } else if (!showTranslation && englishText && ocrResult && devanagariText) {
      // Reset to original Devanagari text when hiding translation
      setOcrResult({
        ...ocrResult,
        text: devanagariText
      })
      setTranslationError(null) // Clear any translation errors
    }
  }, [showTranslation, englishText, devanagariText, translationLoading, ocrResult, handleTranslateToEnglish])
  
  // Clear translation error when new OCR result comes in
  useEffect(() => {
    if (ocrResult && !showTranslation) {
      setTranslationError(null)
      setEnglishText('')
    }
  }, [ocrResult, showTranslation])
  
  // Mobile camera capture
  const handleMobileCapture = async () => {
    const imageSrc = webcamRef.current?.getScreenshot()
    
    if (imageSrc) {
      // Convert base64 to File
      const response = await fetch(imageSrc)
      const blob = await response.blob()
      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })
      
      setPreview(imageSrc)
      setImage(file)
      setCameraActive(false)
      
      // Start OCR processing
      setLoading(true)
      
      try {
        const result = await recognizeText(file, getAuthHeaders(), cookieId)
        handleOCRComplete(result)
      } catch (error) {
        console.error('OCR Error:', error)
        handleOCRComplete({
          text: error.message || 'Error processing image',
          characters: [],
          confidence: 0,
          error: error.message
        })
      }
    }
  }
  
  // Mobile gallery upload
  const handleMobileGalleryUpload = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = async (event) => {
        setPreview(event.target.result)
        setImage(file)
        setCameraActive(false)
        
        // Start OCR processing
        setLoading(true)
        
        try {
          const result = await recognizeText(file, getAuthHeaders(), cookieId)
          handleOCRComplete(result)
        } catch (error) {
          console.error('OCR Error:', error)
          handleOCRComplete({
            text: error.message || 'Error processing image',
            characters: [],
            confidence: 0,
            error: error.message
          })
        }
      }
      reader.readAsDataURL(file)
    }
  }
  
  // Reset to camera
  const resetToCamera = () => {
    setImage(null)
    setPreview(null)
    setOcrResult(null)
    setCameraActive(true)
    setTranslations({})
    setDevanagariText('')
    setEnglishText('')
    setShowTranslation(false)
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  }

  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'environment'
  }

  return (
    <div className="min-h-screen bg-primary-50">
      <main className="container mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-6 md:py-8 max-w-7xl">
        {/* Enhanced Hero Section - Responsive - Hide on mobile if camera/result active */}
        <motion.div 
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className={`text-center mb-8 sm:mb-12 md:mb-16 ${isMobile && (cameraActive || ocrResult) ? 'hidden' : ''}`}
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="inline-block mb-4 sm:mb-6"
          >
            <img 
              src={heroImage} 
              alt="Lipika Logo" 
              className="max-w-full h-auto mx-auto"
              style={{ maxHeight: '400px' }}
            />
          </motion.div>
          
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="text-lg sm:text-xl md:text-2xl lg:text-3xl text-black max-w-4xl mx-auto font-semibold mb-2 sm:mb-3 md:mb-4 px-4"
          >
            Advanced Ranjana Script OCR
          </motion.p>
          
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="text-sm sm:text-base md:text-lg text-black max-w-3xl mx-auto px-4 leading-relaxed"
          >
            Experience AR overlay and real-time translation powered by cutting-edge AI
          </motion.p>

          {/* Feature Pills - Responsive */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="flex flex-wrap justify-center gap-2 sm:gap-3 md:gap-4 mt-4 sm:mt-6 md:mt-8 px-4"
          >
            {['AI-Powered', 'Real-Time', 'AR Overlay', 'Translation'].map((feature, idx) => (
              <span
                key={idx}
                className="px-3 sm:px-4 py-1.5 sm:py-2 bg-white border border-gray-300 rounded-full text-xs sm:text-sm font-semibold text-black shadow-sm hover:shadow-md transition-all duration-200"
              >
                {feature}
              </span>
            ))}
          </motion.div>
        </motion.div>

        {/* Trial Counter - Show for unregistered users */}
        {!isAuthenticated() && (
          <TrialCounter trialInfo={trialInfo} />
        )}

        {/* User Usage Counter - Always show for authenticated free users */}
        {isAuthenticated() && (
          <UserUsageCounter ocrResult={ocrResult} />
        )}

        {/* Mobile Camera Interface - Google Lens Style */}
        {isMobile && cameraActive && !ocrResult && !loading && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="fixed inset-0 z-40 bg-black"
            style={{ top: '60px' }}
          >
            <div className="relative h-full w-full">
              {/* Camera View */}
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={{
                  width: { ideal: 1280 },
                  height: { ideal: 720 },
                  facingMode: "environment"
                }}
                className="w-full h-full object-cover"
                mirrored={true}
              />
              
              {/* Overlay Guide */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-4/5 h-2/3 border-4 border-white/50 rounded-2xl relative">
                  <div className="absolute -top-2 -left-2 w-8 h-8 border-t-4 border-l-4 border-primary-500 rounded-tl-2xl"></div>
                  <div className="absolute -top-2 -right-2 w-8 h-8 border-t-4 border-r-4 border-primary-500 rounded-tr-2xl"></div>
                  <div className="absolute -bottom-2 -left-2 w-8 h-8 border-b-4 border-l-4 border-primary-500 rounded-bl-2xl"></div>
                  <div className="absolute -bottom-2 -right-2 w-8 h-8 border-b-4 border-r-4 border-primary-500 rounded-br-2xl"></div>
                </div>
              </div>
              
              {/* Instructions & Usage Counter */}
              <div className="absolute top-4 left-0 right-0 text-center px-4 space-y-3">
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-black/70 backdrop-blur-md text-white px-4 py-2 rounded-full inline-block"
                >
                  <p className="text-sm font-medium">Position Ranjana text in frame</p>
                </motion.div>
                
                {/* Mobile Usage Counter for Authenticated Users */}
                {isAuthenticated() && ocrResult?.trialInfo && (
                  <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className={`bg-black/70 backdrop-blur-md px-4 py-2 rounded-full inline-block ${
                      ocrResult.trialInfo.remainingTrials <= 3 ? 'border-2 border-orange-400' : 'border-2 border-primary-400'
                    }`}
                  >
                    <p className="text-xs font-bold text-white">
                      Scans: <span className={`text-base ${
                        ocrResult.trialInfo.remainingTrials <= 3 ? 'text-orange-400' : 'text-primary-400'
                      }`}>
                        {ocrResult.trialInfo.remainingTrials}/{ocrResult.trialInfo.maxTrials}
                      </span>
                    </p>
                  </motion.div>
                )}
              </div>
              
              {/* Bottom Action Buttons */}
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/70 to-transparent pb-8 pt-16">
                <div className="flex items-center justify-center gap-8 px-8">
                  {/* Gallery Upload Button */}
                  <motion.button
                    whileTap={{ scale: 0.9 }}
                    onClick={() => fileInputRef.current?.click()}
                    className="w-14 h-14 rounded-full bg-white/20 backdrop-blur-md border-2 border-white/50 flex items-center justify-center hover:bg-white/30 transition-colors"
                  >
                    <FaImage className="text-white text-xl" />
                  </motion.button>
                  
                  {/* Capture Button */}
                  <motion.button
                    whileTap={{ scale: 0.9 }}
                    onClick={handleMobileCapture}
                    className="w-20 h-20 rounded-full bg-white border-4 border-primary-500 flex items-center justify-center shadow-2xl hover:bg-gray-100 transition-colors relative"
                  >
                    <div className="w-16 h-16 rounded-full bg-white border-4 border-primary-500"></div>
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ repeat: Infinity, duration: 2 }}
                      className="absolute inset-0 rounded-full border-4 border-primary-400 opacity-50"
                    ></motion.div>
                  </motion.button>
                  
                  {/* Placeholder for symmetry */}
                  <div className="w-14 h-14"></div>
                </div>
                
                <p className="text-white text-center text-xs mt-4 opacity-80">
                  Tap to capture • Swipe for gallery
                </p>
              </div>
              
              {/* Hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleMobileGalleryUpload}
                className="hidden"
              />
            </div>
          </motion.div>
        )}

        {/* Upload Section with Enhanced Cards - Responsive Grid - Hide on mobile */}
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className={`grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 md:gap-8 mb-6 sm:mb-8 md:mb-12 items-start ${isMobile ? 'hidden' : ''}`}
        >
          <motion.div variants={itemVariants} className="h-fit">
            <ImageUpload 
              onImageUpload={handleImageUpload}
              onProcessing={handleProcessing}
              onOCRComplete={handleOCRComplete}
              authHeaders={getAuthHeaders()}
              cookieId={cookieId}
            />
          </motion.div>
          <motion.div variants={itemVariants} className="h-fit">
            <CameraCapture 
              onImageCapture={handleImageUpload}
              onProcessing={handleProcessing}
              onOCRComplete={handleOCRComplete}
              authHeaders={getAuthHeaders()}
              cookieId={cookieId}
            />
          </motion.div>
        </motion.div>

        {/* Loading State - Responsive */}
        {loading && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card mb-6 sm:mb-8 text-center py-8 sm:py-12 md:py-16"
          >
            <div className="inline-block animate-spin rounded-full h-12 w-12 sm:h-14 sm:w-14 md:h-16 md:w-16 border-4 border-primary border-t-transparent mb-4 sm:mb-6"></div>
            <motion.p 
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="text-base sm:text-lg md:text-xl text-black font-semibold mb-2"
            >
              Processing your image...
            </motion.p>
            <p className="text-xs sm:text-sm text-black">Our AI is analyzing every character</p>
          </motion.div>
        )}

        {/* Results Section - Responsive */}
        {ocrResult && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-4 sm:space-y-6 md:space-y-8"
          >
            {/* Mobile: Back to Camera Button */}
            {isMobile && (
              <motion.button
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                onClick={resetToCamera}
                className="w-full btn-outline flex items-center justify-center gap-2 mb-4"
              >
                <FaCamera className="text-lg" />
                <span>Scan Another Image</span>
              </motion.button>
            )}
            
            <OCRResult 
              text={ocrResult.text} 
              characters={ocrResult.characters}
              confidence={ocrResult.confidence}
              onToggleAR={toggleAR}
              showAR={showAR}
              translations={translations}
              devanagariText={devanagariText}
              englishText={englishText}
              showTranslation={showTranslation}
              setShowTranslation={setShowTranslation}
              translationLoading={translationLoading}
              translationError={translationError}
              onTranslateToEnglish={handleTranslateToEnglish}
            />
            
            {image && showAR && ocrResult.characters && (
              <AROverlay 
                image={image}
                characters={ocrResult.characters}
                showTranslation={showTranslation}
                translations={translations}
              />
            )}
          </motion.div>
        )}

        {/* Enhanced Info Cards - Responsive Grid - Hide on mobile */}
        {!ocrResult && !loading && (
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 md:gap-8 mt-12 sm:mt-16 md:mt-20 ${isMobile ? 'hidden' : ''}`}
          >
            {[
              {
                icon: FaUpload,
                title: 'Capture or Upload',
                description: 'Take a photo with your camera or upload an image containing Ranjana script text. Supports all major image formats.',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600',
                delay: 0.1
              },
              {
                icon: FaSearch,
                title: 'AI Recognition',
                description: 'Our advanced CRNN model identifies individual characters with high accuracy using state-of-the-art deep learning.',
                bgColor: 'bg-secondary-100',
                iconColor: 'text-secondary-600',
                delay: 0.2
              },
              {
                icon: FaEye,
                title: 'AR Overlay',
                description: 'See recognized text highlighted with interactive bounding boxes and confidence scores.',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600',
                delay: 0.3
              }
            ].map((card, index) => {
              const IconComponent = card.icon
              return (
              <motion.div
                key={index}
                variants={itemVariants}
                className="card hover:shadow-2xl transition-all duration-300 relative overflow-hidden group"
              >
                {/* Background Effect on Hover */}
                <div className="absolute inset-0 bg-primary-50 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                
                <div className="relative z-10">
                  {/* Icon with Animated Background */}
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: index * 0.1 + 0.2, type: "spring", stiffness: 200 }}
                    className={`mb-3 sm:mb-4 inline-block p-3 sm:p-4 rounded-xl ${card.bgColor}`}
                  >
                    <IconComponent className={`text-3xl sm:text-4xl md:text-5xl ${card.iconColor}`} />
                  </motion.div>
                  
                  {/* Title */}
                  <motion.h3 
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 + 0.3 }}
                    className="text-lg sm:text-xl font-bold mb-2 sm:mb-3 text-gray-800 group-hover:text-primary-600 transition-colors duration-300"
                  >
                    {card.title}
                  </motion.h3>
                  
                  {/* Description */}
                  <motion.p 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: index * 0.1 + 0.4 }}
                    className="text-sm sm:text-base text-gray-600 leading-relaxed"
                  >
                    {card.description}
                  </motion.p>
                </div>
                
                {/* Decorative Arrow on Hover */}
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  whileHover={{ opacity: 1, x: 0 }}
                  className="absolute top-4 right-4 text-primary-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                >
                  <FaArrowRight />
                </motion.div>
              </motion.div>
              )
            })}
          </motion.div>
        )}
      </main>
    </div>
  )
}

export default Home
