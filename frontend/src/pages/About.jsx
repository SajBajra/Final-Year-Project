import { motion } from 'framer-motion'

const About = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <main className="container mx-auto px-4 py-12 max-w-4xl">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-5xl font-extrabold mb-4 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
            About Lipika
          </h1>
          <p className="text-xl text-gray-700">
            Preserving and digitizing Ranjana script through advanced AI
          </p>
        </motion.div>

        {/* Mission Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card mb-12"
        >
          <h2 className="text-3xl font-bold mb-6 text-gray-800">Our Mission</h2>
          <p className="text-gray-700 leading-relaxed text-lg mb-4">
            <strong>Lipika</strong> (लिपिका) is an advanced Optical Character Recognition (OCR) system 
            designed specifically for the Ranjana script, an ancient and beautiful writing system used 
            in Nepal, Tibet, and other Himalayan regions.
          </p>
          <p className="text-gray-700 leading-relaxed text-lg">
            Our mission is to preserve, digitize, and make accessible historical and modern documents 
            written in Ranjana script using cutting-edge deep learning technology. We combine the power 
            of Convolutional Recurrent Neural Networks (CRNN) with modern web technologies to create 
            a Google Lens-style experience for Ranjana text recognition.
          </p>
        </motion.div>

        {/* What We Do */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card mb-12"
        >
          <h2 className="text-3xl font-bold mb-6 text-gray-800">What We Do</h2>
          <div className="space-y-4">
            <div className="flex items-start space-x-4">
              <div className="text-2xl">🔍</div>
              <div>
                <h3 className="font-bold text-lg mb-2">Character-Level Recognition</h3>
                <p className="text-gray-700">
                  Our AI model recognizes individual Ranjana characters with high accuracy, enabling 
                  precise text extraction and AR visualization.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-4">
              <div className="text-2xl">👓</div>
              <div>
                <h3 className="font-bold text-lg mb-2">AR Overlay Visualization</h3>
                <p className="text-gray-700">
                  Interactive augmented reality overlay shows recognized text with bounding boxes, 
                  similar to Google Lens, making it easy to understand recognition results.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-4">
              <div className="text-2xl">🌐</div>
              <div>
                <h3 className="font-bold text-lg mb-2">Translation Support</h3>
                <p className="text-gray-700">
                  Translate recognized Ranjana text to English and other languages, helping bridge 
                  language barriers and make content accessible to a wider audience.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-4">
              <div className="text-2xl">📱</div>
              <div>
                <h3 className="font-bold text-lg mb-2">Modern Web Interface</h3>
                <p className="text-gray-700">
                  Beautiful, responsive web application that works on desktop, tablet, and mobile 
                  devices, making OCR accessible to everyone.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Technology */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card mb-12"
        >
          <h2 className="text-3xl font-bold mb-6 text-gray-800">Technology</h2>
          <p className="text-gray-700 leading-relaxed mb-4">
            Lipika is built using a modern three-layer architecture:
          </p>
          <ul className="space-y-3 text-gray-700">
            <li className="flex items-start">
              <span className="text-blue-600 mr-2 font-bold">•</span>
              <span><strong>Model Layer (Python):</strong> PyTorch-based CRNN neural network for character recognition, 
              served via Flask REST API</span>
            </li>
            <li className="flex items-start">
              <span className="text-purple-600 mr-2 font-bold">•</span>
              <span><strong>Presenter Layer (Java):</strong> Spring Boot microservices for business logic, 
              API orchestration, and data validation</span>
            </li>
            <li className="flex items-start">
              <span className="text-pink-600 mr-2 font-bold">•</span>
              <span><strong>View Layer (React):</strong> Modern React frontend with Tailwind CSS for 
              intuitive user interface and AR visualization</span>
            </li>
          </ul>
        </motion.div>

        {/* Dataset */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card mb-12"
        >
          <h2 className="text-3xl font-bold mb-6 text-gray-800">Training Data</h2>
          <p className="text-gray-700 leading-relaxed mb-4">
            Our model is trained on a comprehensive dataset:
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-blue-600 mb-1">164,000+</div>
              <p className="text-gray-700">Character Images</p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-purple-600 mb-1">82</div>
              <p className="text-gray-700">Character Classes</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-600 mb-1">131,200</div>
              <p className="text-gray-700">Training Samples</p>
            </div>
            <div className="bg-pink-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-pink-600 mb-1">32,800</div>
              <p className="text-gray-700">Validation Samples</p>
            </div>
          </div>
        </motion.div>

        {/* Future Goals */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card"
        >
          <h2 className="text-3xl font-bold mb-6 text-gray-800">Future Goals</h2>
          <ul className="space-y-3 text-gray-700">
            <li className="flex items-start">
              <span className="text-green-600 mr-2">✓</span>
              <span>Improve model accuracy through additional training data and fine-tuning</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-2">✓</span>
              <span>Expand translation support to more languages</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-2">✓</span>
              <span>Add text-to-speech functionality for accessibility</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-2">✓</span>
              <span>Develop mobile applications for iOS and Android</span>
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-2">✓</span>
              <span>Support batch processing for large document archives</span>
            </li>
          </ul>
        </motion.div>

        {/* Contact */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="text-center mt-12 card"
        >
          <h2 className="text-2xl font-bold mb-4 text-gray-800">Get Involved</h2>
          <p className="text-gray-700 mb-6">
            Lipika is an open-source project. Contributions, feedback, and suggestions are welcome!
          </p>
          <div className="flex justify-center space-x-4">
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-primary"
            >
              View on GitHub
            </a>
            <a 
              href="mailto:contact@lipika.com" 
              className="btn-secondary"
            >
              Contact Us
            </a>
          </div>
        </motion.div>
      </main>
    </div>
  )
}

export default About

