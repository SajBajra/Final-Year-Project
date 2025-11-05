import { useRef, useState } from 'react'
import { recognizeText } from '../services/ocrService'

const ImageUpload = ({ onImageUpload, onProcessing, onOCRComplete }) => {
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
      const result = await recognizeText(file)
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

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4 flex items-center">
        <span className="mr-3">📁</span>
        Upload Image
      </h2>
      
      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all duration-200 ${
          dragActive
            ? 'border-primary-600 bg-gray-100'
            : 'border-gray-300 hover:border-primary-600 hover:bg-gray-50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={openFileDialog}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        {!preview ? (
          <>
            <div className="text-6xl mb-4">📤</div>
            <p className="text-gray-600 font-medium mb-2">
              Drag & drop your image here
            </p>
            <p className="text-sm text-gray-500">
              or click to browse files
            </p>
            <p className="text-xs text-gray-400 mt-4">
              Supports: JPG, PNG, GIF, BMP
            </p>
          </>
        ) : (
          <img
            src={preview}
            alt="Preview"
            className="max-h-64 mx-auto rounded-lg shadow-md"
          />
        )}
      </div>
    </div>
  )
}

export default ImageUpload

