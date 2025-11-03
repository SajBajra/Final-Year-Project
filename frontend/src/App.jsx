import { useState } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import CameraCapture from './components/CameraCapture'
import OCRResult from './components/OCRResult'
import AROverlay from './components/AROverlay'
import Footer from './components/Footer'

function App() {
  const [image, setImage] = useState(null)
  const [ocrResult, setOcrResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showAR, setShowAR] = useState(false)

  const handleImageUpload = (file) => {
    setImage(file)
    setOcrResult(null)
    setShowAR(false)
  }

  const handleOCRComplete = (result) => {
    setOcrResult(result)
    setLoading(false)
  }

  const handleProcessing = () => {
    setLoading(true)
  }

  const toggleAR = () => {
    setShowAR(!showAR)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gradient mb-4">
            Lipika
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Recognize Ranjana script from images with Google Lens-style AR overlay
          </p>
        </div>

        {/* Upload Section */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
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

        {/* Results Section */}
        {loading && (
          <div className="card mb-8 text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Processing your image...</p>
          </div>
        )}

        {ocrResult && !loading && (
          <div className="space-y-6">
            <OCRResult 
              text={ocrResult.text} 
              characters={ocrResult.characters}
              confidence={ocrResult.confidence}
              onToggleAR={toggleAR}
              showAR={showAR}
            />
            
            {image && showAR && ocrResult.characters && (
              <AROverlay 
                image={image}
                characters={ocrResult.characters}
              />
            )}
          </div>
        )}

        {/* Info Cards */}
        {!ocrResult && !loading && (
          <div className="grid md:grid-cols-3 gap-6 mt-12">
            <div className="card text-center">
              <div className="text-4xl mb-3">📸</div>
              <h3 className="font-bold text-lg mb-2">Capture or Upload</h3>
              <p className="text-gray-600 text-sm">
                Take a photo or upload an image containing Ranjana text
              </p>
            </div>
            <div className="card text-center">
              <div className="text-4xl mb-3">🔍</div>
              <h3 className="font-bold text-lg mb-2">AI Recognition</h3>
              <p className="text-gray-600 text-sm">
                Our CRNN model identifies individual characters with high accuracy
              </p>
            </div>
            <div className="card text-center">
              <div className="text-4xl mb-3">👓</div>
              <h3 className="font-bold text-lg mb-2">AR Overlay</h3>
              <p className="text-gray-600 text-sm">
                See recognized text highlighted in Google Lens style
              </p>
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}

export default App

