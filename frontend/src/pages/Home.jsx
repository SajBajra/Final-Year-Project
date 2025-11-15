import { useState } from 'react'
import { motion } from 'framer-motion'
import { FaScroll, FaCamera, FaSearch, FaEye } from 'react-icons/fa'
import ImageUpload from '../components/ImageUpload'
import CameraCapture from '../components/CameraCapture'
import OCRResult from '../components/OCRResult'
import AROverlay from '../components/AROverlay'
import { recognizeText, translateText } from '../services/ocrService'

function Home() {
  const [image, setImage] = useState(null)
  const [ocrResult, setOcrResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showAR, setShowAR] = useState(false)
  const [translations, setTranslations] = useState({})
  const [showTranslation, setShowTranslation] = useState(false)
  const [translationLoading, setTranslationLoading] = useState(false)

  const handleImageUpload = (file) => {
    setImage(file)
    setOcrResult(null)
    setShowAR(false)
    setTranslations({})
  }

  const handleOCRComplete = async (result) => {
    setOcrResult(result)
    setLoading(false)
    
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

  const [devanagariText, setDevanagariText] = useState('')
  const [englishText, setEnglishText] = useState('')

  // Removed handleTranslateToDevanagari - OCR already returns Devanagari

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
    <div className="min-h-screen bg-gray-50">
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Enhanced Hero Section */}
        <motion.div 
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="inline-block mb-6"
          >
            <FaScroll className="text-8xl float-animation text-primary-600" />
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="text-7xl md:text-8xl font-black mb-6 text-primary-600 tracking-tight"
          >
            Lipika
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="text-2xl md:text-3xl text-gray-800 max-w-4xl mx-auto font-semibold mb-4"
          >
            Advanced Ranjana Script OCR
          </motion.p>
          
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="text-lg text-secondary-500 max-w-3xl mx-auto"
          >
            Experience Google Lens-style AR overlay and real-time translation powered by cutting-edge AI
          </motion.p>

          {/* Feature Pills */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="flex flex-wrap justify-center gap-4 mt-8"
          >
            {['AI-Powered', 'Real-Time', 'AR Overlay', 'Translation'].map((feature, idx) => (
              <span
                key={idx}
                className="px-4 py-2 bg-white border border-gray-300 rounded-full text-sm font-semibold text-secondary-500 shadow-sm hover:shadow-md transition-all duration-200"
              >
                {feature}
              </span>
            ))}
          </motion.div>
        </motion.div>

        {/* Upload Section with Enhanced Cards */}
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid md:grid-cols-2 gap-8 mb-12"
        >
          <motion.div variants={itemVariants}>
            <ImageUpload 
              onImageUpload={handleImageUpload}
              onProcessing={handleProcessing}
              onOCRComplete={handleOCRComplete}
            />
          </motion.div>
          <motion.div variants={itemVariants}>
            <CameraCapture 
              onImageCapture={handleImageUpload}
              onProcessing={handleProcessing}
              onOCRComplete={handleOCRComplete}
            />
          </motion.div>
        </motion.div>

        {/* Loading State */}
        {loading && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card mb-8 text-center py-16"
          >
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-primary border-t-transparent mb-6"></div>
            <motion.p 
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="text-xl text-gray-800 font-semibold mb-2"
            >
              Processing your image...
            </motion.p>
            <p className="text-sm text-secondary-500">Our AI is analyzing every character</p>
          </motion.div>
        )}

        {/* Results Section */}
        {ocrResult && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
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

        {/* Enhanced Info Cards */}
        {!ocrResult && !loading && (
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="grid md:grid-cols-3 gap-8 mt-20"
          >
            {[
              {
                icon: FaCamera,
                title: 'Capture or Upload',
                description: 'Take a photo with your camera or upload an image containing Ranjana script text. Supports all major image formats.',
                delay: 0.1
              },
              {
                icon: FaSearch,
                title: 'AI Recognition',
                description: 'Our advanced CRNN model identifies individual characters with high accuracy using state-of-the-art deep learning.',
                delay: 0.2
              },
              {
                icon: FaEye,
                title: 'AR Overlay',
                description: 'See recognized text highlighted in Google Lens style with interactive bounding boxes and confidence scores.',
                delay: 0.3
              }
            ].map((card, index) => {
              const IconComponent = card.icon
              return (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -5 }}
                className="card group cursor-pointer"
              >
                <div className="text-6xl mb-6 inline-block p-4 rounded-xl bg-gray-100 group-hover:bg-gray-200 transition-all duration-200 flex items-center justify-center">
                  <IconComponent className="text-primary-600" />
                </div>
                <h3 className="text-2xl font-bold mb-4 text-gray-800 group-hover:text-primary-600 transition-all duration-200">
                  {card.title}
                </h3>
                <p className="text-secondary-500 leading-relaxed">
                  {card.description}
                </p>
                <div className="mt-6 h-1 w-0 group-hover:w-full bg-primary-600 rounded-full transition-all duration-300"></div>
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
