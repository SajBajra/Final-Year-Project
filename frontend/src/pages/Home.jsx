import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FaScroll, FaCamera, FaSearch, FaEye, FaUpload, FaArrowRight } from 'react-icons/fa'
import heroImage from '../images/HeroSection-FrontPage.png'
import { useAuth } from '../context/AuthContext'
import { getOrCreateTrialCookie } from '../utils/cookieUtils'
import ImageUpload from '../components/ImageUpload'
import CameraCapture from '../components/CameraCapture'
import OCRResult from '../components/OCRResult'
import AROverlay from '../components/AROverlay'
import TrialCounter from '../components/TrialCounter'
import { recognizeText, translateText } from '../services/ocrService'

function Home() {
  const [image, setImage] = useState(null)
  const [ocrResult, setOcrResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showAR, setShowAR] = useState(false)
  const [translations, setTranslations] = useState({})
  const [showTranslation, setShowTranslation] = useState(false)
  const [translationLoading, setTranslationLoading] = useState(false)
  const [devanagariText, setDevanagariText] = useState('')
  const [englishText, setEnglishText] = useState('')
  const [trialInfo, setTrialInfo] = useState(null)
  
  const { isAuthenticated, getAuthHeaders } = useAuth()
  const cookieId = getOrCreateTrialCookie()
  
  useEffect(() => {
    // Set cookie if not already set
    if (!document.cookie.includes('lipika_trial_id')) {
      getOrCreateTrialCookie()
    }
  }, [])

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
    
    // Check if trial limit exceeded
    if (result?.error && result?.trialInfo?.requiresLogin) {
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
    }
  }

  const handleProcessing = () => {
    setLoading(true)
  }

  const toggleAR = () => {
    setShowAR(!showAR)
  }

  const handleTranslateToEnglish = async () => {
    setTranslationLoading(true)
    
    try {
      // Translate Devanagari to English via API
      const textToTranslate = devanagariText || ocrResult?.text || ''
      const englishResult = await translateText(textToTranslate, 'en')
      if (englishResult) {
        setEnglishText(englishResult)
        setTranslations({ ...translations, english: englishResult })
      }
    } catch (error) {
      console.error('English translation error:', error)
    } finally {
      setTranslationLoading(false)
    }
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

  return (
    <div className="min-h-screen bg-primary-50">
      <main className="container mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-6 md:py-8 max-w-7xl">
        {/* Enhanced Hero Section - Responsive */}
        <motion.div 
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8 sm:mb-12 md:mb-16"
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

        {/* Upload Section with Enhanced Cards - Responsive Grid */}
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 md:gap-8 mb-6 sm:mb-8 md:mb-12"
        >
          <motion.div variants={itemVariants}>
            <ImageUpload 
              onImageUpload={handleImageUpload}
              onProcessing={handleProcessing}
              onOCRComplete={handleOCRComplete}
              authHeaders={getAuthHeaders()}
              cookieId={cookieId}
            />
          </motion.div>
          <motion.div variants={itemVariants}>
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

        {/* Enhanced Info Cards - Responsive Grid */}
        {!ocrResult && !loading && (
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 md:gap-8 mt-12 sm:mt-16 md:mt-20"
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
