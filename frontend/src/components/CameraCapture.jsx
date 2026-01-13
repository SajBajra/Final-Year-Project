import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import Webcam from 'react-webcam'
import { FaCamera, FaCameraRetro, FaTimes } from 'react-icons/fa'
import { recognizeText } from '../services/ocrService'

const CameraCapture = ({ onImageCapture, onProcessing, onOCRComplete, authHeaders = {}, cookieId = null }) => {
  const webcamRef = useRef(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [preview, setPreview] = useState(null)
  const [error, setError] = useState(null)

  const videoConstraints = {
    width: { min: 640, ideal: 1280, max: 1920 },
    height: { min: 480, ideal: 720, max: 1080 },
    facingMode: 'user', // Front camera for both mobile and desktop
    aspectRatio: { ideal: 16/9 }
  }

  const startCapture = () => {
    setIsCapturing(true)
    setPreview(null)
    setError(null)
  }

  const stopCapture = () => {
    setIsCapturing(false)
    setError(null)
  }

  const handleUserMediaError = (error) => {
    console.error('Camera Error:', error)
    setIsCapturing(false)
    
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      setError('Camera permission denied. Please allow camera access in your browser settings.')
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      setError('No camera found. Please connect a camera and try again.')
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      setError('Camera is already in use by another application.')
    } else if (error.name === 'OverconstrainedError') {
      setError('Camera does not support the requested settings.')
    } else if (error.name === 'TypeError') {
      setError('Camera access requires HTTPS. Please use a secure connection.')
    } else {
      setError(`Camera error: ${error.message || 'Unable to access camera'}`)
    }
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
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-4 bg-red-50 border border-red-200 rounded-xl"
          >
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-600" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-red-800">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="flex-shrink-0 text-red-600 hover:text-red-800"
              >
                <FaTimes />
              </button>
            </div>
          </motion.div>
        )}
        
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
            className="aspect-video bg-black rounded-xl overflow-hidden relative shadow-xl"
          >
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="w-full h-full object-cover"
              mirrored={true}
              onUserMediaError={handleUserMediaError}
            />
            <div className="absolute inset-0 border-4 border-primary-600 rounded-xl pointer-events-none"></div>
            
            {/* Overlay Camera Controls */}
            <div className="absolute bottom-6 left-0 right-0 flex items-center justify-center gap-6 z-10">
              {/* Cancel Button */}
              <motion.button
                onClick={stopCapture}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-red-500 hover:bg-red-600 flex items-center justify-center shadow-lg hover:shadow-xl transition-all duration-200"
                title="Cancel"
              >
                <FaTimes className="text-white text-xl sm:text-2xl" />
              </motion.button>
              
              {/* Capture Button */}
              <motion.button
                onClick={capture}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-primary-600 hover:bg-primary-700 flex items-center justify-center shadow-lg hover:shadow-xl transition-all duration-200 border-4 border-white"
                title="Capture"
              >
                <FaCamera className="text-white text-2xl sm:text-3xl" />
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
