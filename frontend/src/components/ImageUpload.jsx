import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { FaUpload, FaFileImage } from 'react-icons/fa'
import { recognizeText } from '../services/ocrService'

const ImageUpload = ({ onImageUpload, onProcessing, onOCRComplete, authHeaders = {}, cookieId = null }) => {
  const fileInputRef = useRef(null)
  const [dragActive, setDragActive] = useState(false)
  const [preview, setPreview] = useState(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = async (file) => {
    // Show preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target.result)
    }
    reader.readAsDataURL(file)
    
    // Call parent to update image
    onImageUpload(file)
    
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
  }

  const openFileDialog = () => {
    fileInputRef.current?.click()
  }

  const clearPreview = () => {
    setPreview(null)
    onImageUpload(null)
  }

  return (
    <div className="card h-full">
      <div className="flex items-center space-x-2 sm:space-x-3 mb-4 sm:mb-6">
        <div className="p-2 sm:p-3 rounded-xl bg-primary-100">
          <FaUpload className="text-xl sm:text-2xl text-primary-600" />
        </div>
        <h2 className="text-xl sm:text-2xl font-bold text-gray-800">
          Upload Image
        </h2>
      </div>
      
      <div
        className={`border-2 border-dashed rounded-xl p-6 sm:p-8 md:p-12 text-center cursor-pointer transition-all duration-200 ${
          dragActive
            ? 'border-primary-600 bg-primary-50'
            : 'border-gray-300 hover:border-primary-600 hover:bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={!preview ? openFileDialog : undefined}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        {!preview ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3 sm:space-y-4"
          >
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{ repeat: Infinity, duration: 2 }}
              className="flex justify-center"
            >
              <FaUpload className="text-4xl sm:text-5xl md:text-6xl text-primary-600" />
            </motion.div>
            <div>
              <p className="text-base sm:text-lg md:text-xl text-gray-700 font-semibold mb-1 sm:mb-2">
                Drag & drop your image here
              </p>
              <p className="text-xs sm:text-sm text-gray-500 mb-2 sm:mb-3">
                or click to browse files
              </p>
              <p className="text-xs text-gray-400">
                Supports: JPG, PNG, GIF, BMP, WEBP
              </p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative"
          >
            <img
              src={preview}
              alt="Preview"
              className="max-h-48 sm:max-h-56 md:max-h-64 lg:max-h-72 mx-auto rounded-lg shadow-lg object-contain w-full"
            />
            <motion.button
              onClick={(e) => {
                e.stopPropagation()
                clearPreview()
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="mt-3 sm:mt-4 px-4 sm:px-6 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg font-medium transition-all duration-200 text-sm sm:text-base"
            >
              Remove Image
            </motion.button>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default ImageUpload
