import { useRef, useState } from 'react'
import Webcam from 'react-webcam'
import { recognizeText } from '../services/ocrService'

const CameraCapture = ({ onImageCapture, onProcessing, onOCRComplete }) => {
  const webcamRef = useRef(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [preview, setPreview] = useState(null)

  const videoConstraints = {
    width: 1280,
    height: 720,
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
        const result = await recognizeText(file)
        onOCRComplete(result)
      } catch (error) {
        console.error('OCR Error:', error)
        onOCRComplete({
          text: 'Error processing image',
          characters: [],
          confidence: 0
        })
      }
      
      setIsCapturing(false)
    }
  }

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4 flex items-center">
        <span className="mr-3">📷</span>
        Camera Capture
      </h2>
      
      <div className="relative">
        {!isCapturing && !preview && (
          <div className="aspect-video bg-gray-200 rounded-xl flex items-center justify-center">
            <button
              onClick={startCapture}
              className="btn-primary text-lg px-8"
            >
              Start Camera
            </button>
          </div>
        )}
        
        {isCapturing && !preview && (
          <>
            <div className="aspect-video bg-black rounded-xl overflow-hidden">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="flex justify-center space-x-4 mt-4">
              <button
                onClick={capture}
                className="btn-primary"
              >
                📸 Capture
              </button>
              <button
                onClick={stopCapture}
                className="btn-secondary"
              >
                Cancel
              </button>
            </div>
          </>
        )}
        
        {preview && (
          <div>
            <img
              src={preview}
              alt="Captured"
              className="aspect-video w-full object-cover rounded-xl"
            />
            <button
              onClick={() => {
                setPreview(null)
                onImageCapture(null)
              }}
              className="btn-secondary w-full mt-4"
            >
              Capture Again
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default CameraCapture

