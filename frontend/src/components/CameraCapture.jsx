import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import Webcam from 'react-webcam'
import { FaCamera, FaCameraRetro, FaTimes } from 'react-icons/fa'
import { recognizeText } from '../services/ocrService'

const CameraCapture = ({ onImageCapture, onProcessing, onOCRComplete, authHeaders = {}, cookieId = null }) => {
  const webcamRef = useRef(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [preview, setPreview] = useState(null)

  const videoConstraints = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'environment'
  }

  const startCapture = () => {
    setIsCapturing(true)
    setPreview(null)
  }

  const stopCapture = () => {
    setIsCapturing(false)
  }

  const capture = async () => {
    const imageSrc = webcamRef.current?.getScreenshot()
    
    if (imageSrc) {
      // Convert base64 to File
      const response = await fetch(imageSrc)
      const blob = await response.blob()
      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' })
      
      setPreview(imageSrc)
      onImageCapture(file)
      
      // Start OCR processing
      onProcessing()
      
      try {
        const result = await recognizeText(file, authHeaders, cookieId)
        onOCRComplete(result)
      } catch (error) {
        console.error('OCR Error:', error)
        // Show actual error message instead of generic "Error processing image"
        onOCRComplete({
          text: error.message || 'Error processing image',
          characters: [],
          confidence: 0,
          error: error.message
        })
      }
      
      setIsCapturing(false)
    }
  }

  const clearPreview = () => {
    setPreview(null)
    setIsCapturing(false)
    onImageCapture(null)
  }

  return (
    <div className="card h-full">
      <div className="flex items-center space-x-2 sm:space-x-3 mb-4 sm:mb-6">
        <div className="p-2 sm:p-3 rounded-xl bg-secondary-100">
          <FaCamera className="text-xl sm:text-2xl text-secondary-600" />
        </div>
        <h2 className="text-xl sm:text-2xl font-bold text-gray-800">
          Camera Capture
        </h2>
      </div>
      
      <div className="relative">
        {!isCapturing && !preview && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="aspect-video bg-gray-100 rounded-xl flex flex-col items-center justify-center p-6 sm:p-8"
          >
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ repeat: Infinity, duration: 2 }}
            >
              <FaCameraRetro className="text-5xl sm:text-6xl md:text-7xl text-secondary-500 mb-4 sm:mb-6" />
            </motion.div>
            <motion.button
              onClick={startCapture}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn-primary text-sm sm:text-base px-6 sm:px-8 py-3 sm:py-4 flex items-center space-x-2"
            >
              <FaCamera className="text-lg sm:text-xl" />
              <span>Start Camera</span>
            </motion.button>
            <p className="text-xs sm:text-sm text-gray-500 mt-3 sm:mt-4 text-center">
              Click to activate your camera
            </p>
          </motion.div>
        )}
        
        {isCapturing && !preview && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-4"
          >
            <div className="aspect-video bg-black rounded-xl overflow-hidden relative shadow-xl">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 border-4 border-primary-600 rounded-xl pointer-events-none"></div>
            </div>
            <div className="flex flex-col sm:flex-row justify-center gap-3 sm:gap-4">
              <motion.button
                onClick={capture}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-primary flex-1 sm:flex-none flex items-center justify-center space-x-2 px-6 sm:px-8 py-3"
              >
                <FaCamera className="text-lg sm:text-xl" />
                <span className="text-sm sm:text-base">Capture</span>
              </motion.button>
              <motion.button
                onClick={stopCapture}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-secondary flex-1 sm:flex-none px-6 sm:px-8 py-3 text-sm sm:text-base"
              >
                Cancel
              </motion.button>
            </div>
          </motion.div>
        )}
        
        {preview && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-3 sm:space-y-4"
          >
            <div className="relative aspect-video rounded-xl overflow-hidden shadow-xl">
              <img
                src={preview}
                alt="Captured"
                className="w-full h-full object-cover"
              />
            </div>
            <motion.button
              onClick={clearPreview}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="btn-secondary w-full flex items-center justify-center space-x-2 py-3 text-sm sm:text-base"
            >
              <FaTimes className="text-base sm:text-lg" />
              <span>Capture Again</span>
            </motion.button>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default CameraCapture
