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
    try {
      const charMap = {}
      if (characters) {
        // Translate individual characters
        for (const char of characters) {
          if (char.character && !charMap[char.character]) {
            const translation = await translateText(char.character)
            charMap[char.character] = translation
          }
        }
      }
      // Also translate full text
      if (text) {
        const fullTranslation = await translateText(text)
        setTranslations({ ...charMap, '_full': fullTranslation })
      } else {
        setTranslations(charMap)
      }
    } catch (error) {
      console.error('Translation error:', error)
    } finally {
      setTranslationLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-6xl font-extrabold mb-4 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent"
          >
            Lipika
          </motion.h1>
          <p className="text-xl text-gray-700 max-w-3xl mx-auto font-medium">
            Advanced Ranjana Script OCR with Google Lens-style AR overlay and real-time translation
          </p>
        </div>

        {/* Upload Section */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          <ImageUpload 
            onImageUpload={handleImageUpload}
            onProcessing={handleProcessing}
            onOCRComplete={handleOCRComplete}
          />
          <CameraCapture 
            onImageCapture={handleImageUpload}
            onProcessing={handleProcessing}
            onOCRComplete={handleOCRComplete}
          />
        </div>

        {/* Loading State */}
        {loading && (
          <div className="card mb-8 text-center py-12">
            <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-blue-600 border-t-transparent mb-4"></div>
            <p className="text-lg text-gray-700 font-medium">Processing your image...</p>
            <p className="text-sm text-gray-500 mt-2">This may take a few seconds</p>
          </div>
        )}

        {/* Results Section */}
        {ocrResult && !loading && (
          <div className="space-y-8">
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
          </div>
        )}

        {/* Info Cards */}
        {!ocrResult && !loading && (
          <div className="grid md:grid-cols-3 gap-8 mt-16">
            <div className="card text-center hover:shadow-xl transition-shadow transform hover:-translate-y-2">
              <div className="text-5xl mb-4">📸</div>
              <h3 className="font-bold text-xl mb-3 text-gray-800">Capture or Upload</h3>
              <p className="text-gray-600">
                Take a photo with your camera or upload an image containing Ranjana script text
              </p>
            </div>
            <div className="card text-center hover:shadow-xl transition-shadow transform hover:-translate-y-2">
              <div className="text-5xl mb-4">🔍</div>
              <h3 className="font-bold text-xl mb-3 text-gray-800">AI Recognition</h3>
              <p className="text-gray-600">
                Our advanced CRNN model identifies individual characters with high accuracy using deep learning
              </p>
            </div>
            <div className="card text-center hover:shadow-xl transition-shadow transform hover:-translate-y-2">
              <div className="text-5xl mb-4">👓</div>
              <h3 className="font-bold text-xl mb-3 text-gray-800">AR Overlay</h3>
              <p className="text-gray-600">
                See recognized text highlighted in Google Lens style with interactive bounding boxes
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default Home
