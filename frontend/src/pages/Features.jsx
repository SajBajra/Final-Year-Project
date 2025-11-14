import { motion } from 'framer-motion'
import { 
  FaCamera, FaRobot, FaEye, FaGlobe, FaBolt, FaBullseye, 
  FaChartBar, FaMobileAlt, FaLock, FaPython, FaCoffee, FaReact 
} from 'react-icons/fa'

const Features = () => {
  const features = [
    {
      icon: FaCamera,
      title: 'Multiple Input Methods',
      description: 'Upload images from your device or capture in real-time using your camera. Supports JPG, PNG, BMP formats.',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: FaRobot,
      title: 'Advanced AI Recognition',
      description: 'Powered by Character-based CRNN (CNN + RNN) deep learning model for accurate character-level recognition.',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: FaEye,
      title: 'Google Lens-Style AR Overlay',
      description: 'Interactive augmented reality visualization with bounding boxes, confidence scores, and hover tooltips.',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: FaGlobe,
      title: 'Real-Time Translation',
      description: 'Translate recognized Ranjana text to English and other languages with character-level translation support.',
      color: 'from-orange-500 to-red-500'
    },
    {
      icon: FaBolt,
      title: 'Fast Processing',
      description: 'Optimized inference pipeline with GPU acceleration support. Process images in under 2 seconds.',
      color: 'from-indigo-500 to-purple-500'
    },
    {
      icon: FaBullseye,
      title: 'Character-Level Precision',
      description: 'Individual character detection with bounding boxes, perfect for AR visualization and translation.',
      color: 'from-pink-500 to-rose-500'
    },
    {
      icon: FaChartBar,
      title: 'Confidence Scoring',
      description: 'Each recognized character comes with a confidence score to help identify potential recognition errors.',
      color: 'from-teal-500 to-cyan-500'
    },
    {
      icon: FaMobileAlt,
      title: 'Responsive Design',
      description: 'Beautiful, modern UI that works seamlessly on desktop, tablet, and mobile devices.',
      color: 'from-yellow-500 to-orange-500'
    },
    {
      icon: FaLock,
      title: 'Secure & Private',
      description: 'All processing happens on your device or private server. Your images are never stored permanently.',
      color: 'from-gray-600 to-gray-800'
    }
  ]

  const techStack = [
    { name: 'React 18', description: 'Modern UI library' },
    { name: 'PyTorch', description: 'Deep learning framework' },
    { name: 'Flask', description: 'REST API backend' },
    { name: 'OpenCV', description: 'Image processing' },
    { name: 'Tailwind CSS', description: 'Styling framework' },
    { name: 'Spring Boot', description: 'Java microservices' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <main className="container mx-auto px-4 py-12 max-w-7xl">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl font-extrabold mb-4 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
            Features
          </h1>
          <p className="text-xl text-gray-700 max-w-3xl mx-auto">
            Discover the powerful capabilities of Lipika Ranjana OCR System
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => {
            const IconComponent = feature.icon
            return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2"
            >
              <div className={`mb-4 bg-gradient-to-r ${feature.color} bg-clip-text text-transparent`}>
                <IconComponent className="text-5xl" />
              </div>
              <h3 className="text-xl font-bold mb-3 text-gray-800">{feature.title}</h3>
              <p className="text-gray-600 leading-relaxed">{feature.description}</p>
            </motion.div>
            )
          })}
        </div>

        {/* Architecture Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card bg-gradient-to-br from-gray-900 to-gray-800 text-white mb-16"
        >
          <h2 className="text-3xl font-bold mb-8 text-center">System Architecture</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="text-4xl mb-4 flex justify-center">
                <FaPython className="text-yellow-400" />
              </div>
              <h3 className="text-xl font-bold mb-2">Model Layer</h3>
              <p className="text-gray-300">Python-based CRNN model with Flask API for character recognition</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-4 flex justify-center">
                <FaCoffee className="text-orange-400" />
              </div>
              <h3 className="text-xl font-bold mb-2">Presenter Layer</h3>
              <p className="text-gray-300">Java Spring Boot microservices for business logic and orchestration</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-4 flex justify-center">
                <FaReact className="text-cyan-400" />
              </div>
              <h3 className="text-xl font-bold mb-2">View Layer</h3>
              <p className="text-gray-300">React frontend with modern UI for user interaction</p>
            </div>
          </div>
        </motion.div>

        {/* Technology Stack */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <h2 className="text-3xl font-bold mb-8 text-center text-gray-800">Technology Stack</h2>
          <div className="grid md:grid-cols-3 gap-6">
            {techStack.map((tech, index) => (
              <div key={index} className="text-center p-4 bg-gray-50 rounded-lg">
                <h3 className="font-bold text-lg text-gray-800 mb-1">{tech.name}</h3>
                <p className="text-sm text-gray-600">{tech.description}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Performance Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid md:grid-cols-4 gap-6 mt-16"
        >
          <div className="card text-center">
            <div className="text-4xl font-bold text-blue-600 mb-2">&lt;2s</div>
            <p className="text-gray-600">Processing Time</p>
          </div>
          <div className="card text-center">
            <div className="text-4xl font-bold text-green-600 mb-2">82</div>
            <p className="text-gray-600">Character Classes</p>
          </div>
          <div className="card text-center">
            <div className="text-4xl font-bold text-purple-600 mb-2">90%+</div>
            <p className="text-gray-600">Accuracy</p>
          </div>
          <div className="card text-center">
            <div className="text-4xl font-bold text-pink-600 mb-2">164K</div>
            <p className="text-gray-600">Training Images</p>
          </div>
        </motion.div>
      </main>
    </div>
  )
}

export default Features

