import { useState } from 'react'
import { motion } from 'framer-motion'
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
    
    // Auto-translate if characters are detected
    if (result.characters && result.characters.length > 0) {
      handleTranslate(result.text, result.characters)
    }
  }

  const handleProcessing = () => {
    setLoading(true)
  }

  const toggleAR = () => {
    setShowAR(!showAR)
  }

  const handleTranslate = async (text, characters) => {
    setTranslationLoading(true)
    setShowTranslation(true)
    
    try {
      const translations_map = {}
      for (const char of characters) {
        if (char.character && !translations_map[char.character]) {
          try {
            const result = await translateText(char.character, 'en')
            if (result && result.translatedText) {
              translations_map[char.character] = result.translatedText
            }
          } catch (e) {
            console.error(`Translation error for ${char.character}:`, e)
          }
        }
      }
      setTranslations(translations_map)
    } catch (error) {
      console.error('Translation error:', error)
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 via-purple-50 to-pink-50 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute top-40 right-10 w-72 h-72 bg-purple-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-1/2 w-72 h-72 bg-pink-400 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      <main className="container mx-auto px-4 py-8 max-w-7xl relative z-10">
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
            <div className="text-8xl float-animation">📜</div>
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="text-7xl md:text-8xl font-black mb-6 text-gradient bg-clip-text text-transparent text-shadow-lg tracking-tight"
          >
            Lipika
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="text-2xl md:text-3xl text-gray-700 max-w-4xl mx-auto font-semibold mb-4"
          >
            Advanced Ranjana Script OCR
          </motion.p>
          
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="text-lg text-gray-600 max-w-3xl mx-auto"
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
                className="px-4 py-2 bg-white/60 backdrop-blur-sm border border-white/40 rounded-full text-sm font-semibold text-gray-700 shadow-md hover:shadow-lg transition-all duration-300 hover:scale-105"
              >
                ✨ {feature}
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

        {/* Enhanced Loading State */}
        {loading && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card-glow mb-8 text-center py-16"
          >
            <div className="relative inline-block">
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 opacity-75 blur-xl animate-pulse"></div>
              <div className="relative inline-block animate-spin rounded-full h-20 w-20 border-4 border-transparent bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-border">
                <div className="absolute inset-2 rounded-full bg-white"></div>
              </div>
            </div>
            <motion.p 
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="text-xl text-gray-700 font-semibold mt-6 mb-2"
            >
              Processing your image...
            </motion.p>
            <p className="text-sm text-gray-500">Our AI is analyzing every character</p>
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
              showTranslation={showTranslation}
              setShowTranslation={setShowTranslation}
              translationLoading={translationLoading}
              onTranslate={() => handleTranslate(ocrResult.text, ocrResult.characters)}
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
                icon: '📸',
                title: 'Capture or Upload',
                description: 'Take a photo with your camera or upload an image containing Ranjana script text. Supports all major image formats.',
                gradient: 'from-blue-500 to-cyan-500',
                delay: 0.1
              },
              {
                icon: '🔍',
                title: 'AI Recognition',
                description: 'Our advanced CRNN model identifies individual characters with high accuracy using state-of-the-art deep learning.',
                gradient: 'from-purple-500 to-pink-500',
                delay: 0.2
              },
              {
                icon: '👓',
                title: 'AR Overlay',
                description: 'See recognized text highlighted in Google Lens style with interactive bounding boxes and confidence scores.',
                gradient: 'from-pink-500 to-orange-500',
                delay: 0.3
              }
            ].map((card, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ y: -10, scale: 1.02 }}
                className="card-glow group cursor-pointer"
              >
                <div className={`text-6xl mb-6 inline-block p-4 rounded-2xl bg-gradient-to-br ${card.gradient} bg-opacity-10 group-hover:bg-opacity-20 transition-all duration-300`}>
                  {card.icon}
                </div>
                <h3 className="text-2xl font-bold mb-4 text-gray-800 group-hover:text-gradient transition-all duration-300">
                  {card.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {card.description}
                </p>
                <div className={`mt-6 h-1 w-0 group-hover:w-full bg-gradient-to-r ${card.gradient} rounded-full transition-all duration-500`}></div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </main>

      <style jsx>{`
        @keyframes blob {
          0% {
            transform: translate(0px, 0px) scale(1);
          }
          33% {
            transform: translate(30px, -50px) scale(1.1);
          }
          66% {
            transform: translate(-20px, 20px) scale(0.9);
          }
          100% {
            transform: translate(0px, 0px) scale(1);
          }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </div>
  )
}

export default Home
