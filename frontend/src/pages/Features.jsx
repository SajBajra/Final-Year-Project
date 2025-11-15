import { motion } from 'framer-motion'
import { 
  FaCamera, FaRobot, FaEye, FaGlobe, FaBolt, FaBullseye, 
  FaChartBar, FaMobileAlt, FaLock, FaPython, FaCoffee, FaReact, FaUpload
} from 'react-icons/fa'

const Features = () => {
  const features = [
    {
      icon: FaUpload,
      title: 'Multiple Input Methods',
      description: 'Upload images from your device or capture in real-time using your camera. Supports JPG, PNG, WEBP, and BMP formats for maximum flexibility.',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: FaRobot,
      title: 'Advanced AI Recognition',
      description: 'Powered by Character-based CRNN (Convolutional Recurrent Neural Network) deep learning model trained on 500 epochs for accurate character-level recognition of Ranjana script.',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: FaEye,
      title: 'Google Lens-Style AR Overlay',
      description: 'Interactive augmented reality visualization with individual character bounding boxes, confidence scores, and hover tooltips for detailed recognition insights.',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: FaGlobe,
      title: 'Ranjana to Devanagari Output',
      description: 'Automatically converts recognized Ranjana script characters to Devanagari (Nepali) text output, with optional English translation support.',
      color: 'from-orange-500 to-red-500'
    },
    {
      icon: FaBolt,
      title: 'Fast Processing',
      description: 'Optimized inference pipeline with GPU acceleration support. Process images with character segmentation and recognition in seconds.',
      color: 'from-indigo-500 to-purple-500'
    },
    {
      icon: FaBullseye,
      title: 'Character-Level Precision',
      description: 'Individual character detection with precise bounding boxes, confidence scores, and word grouping for perfect AR visualization.',
      color: 'from-pink-500 to-rose-500'
    },
    {
      icon: FaChartBar,
      title: 'Confidence Scoring',
      description: 'Each recognized character includes a confidence score (0-100%) with color-coded visualization to help identify recognition accuracy.',
      color: 'from-teal-500 to-cyan-500'
    },
    {
      icon: FaMobileAlt,
      title: 'Fully Responsive Design',
      description: 'Beautiful, modern UI that works seamlessly on desktop, tablet, and mobile devices with optimized layouts for every screen size.',
      color: 'from-yellow-500 to-orange-500'
    },
    {
      icon: FaLock,
      title: 'Secure & Private',
      description: 'All processing happens on your server. Images are processed in real-time and OCR history is stored securely with admin dashboard access.',
      color: 'from-gray-600 to-gray-800'
    }
  ]

  const techStack = [
    { name: 'React 18 + Vite', description: 'Modern UI framework with fast build' },
    { name: 'PyTorch', description: 'Deep learning framework for CRNN model' },
    { name: 'Flask', description: 'Python REST API for OCR service' },
    { name: 'OpenCV', description: 'Advanced image processing and segmentation' },
    { name: 'Tailwind CSS', description: 'Utility-first CSS framework' },
    { name: 'Spring Boot', description: 'Java backend for API orchestration' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
      <main className="container mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-6 sm:py-8 md:py-12 max-w-7xl">
        {/* Header - Responsive */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8 sm:mb-12 md:mb-16"
        >
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold mb-3 sm:mb-4 bg-gradient-to-r from-primary-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
            Features
          </h1>
          <p className="text-base sm:text-lg md:text-xl text-gray-700 max-w-3xl mx-auto px-4">
            Discover the powerful capabilities of Lipika - Advanced Ranjana Script OCR System
          </p>
        </motion.div>

        {/* Features Grid - Responsive */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 md:gap-8 mb-8 sm:mb-12 md:mb-16">
          {features.map((feature, index) => {
            const IconComponent = feature.icon
            return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -5 }}
              className="card hover:shadow-2xl transition-all duration-300"
            >
              <div className={`mb-3 sm:mb-4 inline-block p-3 sm:p-4 rounded-xl bg-gradient-to-r ${feature.color} bg-opacity-10`}>
                <IconComponent className={`text-3xl sm:text-4xl md:text-5xl bg-gradient-to-r ${feature.color} bg-clip-text text-transparent`} />
              </div>
              <h3 className="text-lg sm:text-xl font-bold mb-2 sm:mb-3 text-gray-800">{feature.title}</h3>
              <p className="text-sm sm:text-base text-gray-600 leading-relaxed">{feature.description}</p>
            </motion.div>
            )
          })}
        </div>

        {/* Architecture Section - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card bg-gradient-to-br from-gray-900 to-gray-800 text-white mb-8 sm:mb-12 md:mb-16"
        >
          <h2 className="text-2xl sm:text-3xl font-bold mb-6 sm:mb-8 text-center">System Architecture</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8">
            <div className="text-center">
              <div className="text-3xl sm:text-4xl mb-3 sm:mb-4 flex justify-center">
                <FaPython className="text-yellow-400" />
              </div>
              <h3 className="text-lg sm:text-xl font-bold mb-2">Model Layer</h3>
              <p className="text-sm sm:text-base text-gray-300">Python-based CRNN model with Flask REST API for character recognition from Ranjana script images</p>
            </div>
            <div className="text-center">
              <div className="text-3xl sm:text-4xl mb-3 sm:mb-4 flex justify-center">
                <FaCoffee className="text-orange-400" />
              </div>
              <h3 className="text-lg sm:text-xl font-bold mb-2">Presenter Layer</h3>
              <p className="text-sm sm:text-base text-gray-300">Java Spring Boot microservices for business logic, API orchestration, and admin dashboard</p>
            </div>
            <div className="text-center">
              <div className="text-3xl sm:text-4xl mb-3 sm:mb-4 flex justify-center">
                <FaReact className="text-cyan-400" />
              </div>
              <h3 className="text-lg sm:text-xl font-bold mb-2">View Layer</h3>
              <p className="text-sm sm:text-base text-gray-300">React frontend with responsive design for intuitive user interface and AR visualization</p>
            </div>
          </div>
        </motion.div>

        {/* Technology Stack - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card mb-8 sm:mb-12 md:mb-16"
        >
          <h2 className="text-2xl sm:text-3xl font-bold mb-6 sm:mb-8 text-center text-gray-800">Technology Stack</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4 md:gap-6">
            {techStack.map((tech, index) => (
              <div key={index} className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <h3 className="font-bold text-sm sm:text-base md:text-lg text-gray-800 mb-1">{tech.name}</h3>
                <p className="text-xs sm:text-sm text-gray-600">{tech.description}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Performance Metrics - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 md:gap-6"
        >
          <div className="card text-center">
            <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-primary-600 mb-1 sm:mb-2">&lt;2s</div>
            <p className="text-xs sm:text-sm text-gray-600">Processing Time</p>
          </div>
          <div className="card text-center">
            <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-green-600 mb-1 sm:mb-2">62</div>
            <p className="text-xs sm:text-sm text-gray-600">Devanagari Characters</p>
          </div>
          <div className="card text-center">
            <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-purple-600 mb-1 sm:mb-2">500</div>
            <p className="text-xs sm:text-sm text-gray-600">Training Epochs</p>
          </div>
          <div className="card text-center">
            <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-pink-600 mb-1 sm:mb-2">90%+</div>
            <p className="text-xs sm:text-sm text-gray-600">Model Accuracy</p>
          </div>
        </motion.div>
      </main>
    </div>
  )
}

export default Features
