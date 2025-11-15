import { motion } from 'framer-motion'
import { FaSearch, FaEye, FaGlobe, FaMobileAlt, FaPython, FaJava, FaReact, FaDatabase, FaChartLine } from 'react-icons/fa'

const About = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100">
      <main className="container mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-6 sm:py-8 md:py-12 max-w-4xl">
        {/* Header - Responsive */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8 sm:mb-12 md:mb-16"
        >
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold mb-3 sm:mb-4 bg-gradient-to-r from-primary-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
            About Lipika
          </h1>
          <p className="text-base sm:text-lg md:text-xl text-gray-700 px-4">
            Preserving and digitizing Ranjana script through advanced AI technology
          </p>
        </motion.div>

        {/* Mission Section - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <h2 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-gray-800">Our Mission</h2>
          <div className="space-y-4">
            <p className="text-base sm:text-lg text-gray-700 leading-relaxed">
              <strong>Lipika</strong> (लिपिका) is an advanced Optical Character Recognition (OCR) system 
              designed specifically for the Ranjana script, an ancient and beautiful writing system used 
              in Nepal, Tibet, and other Himalayan regions.
            </p>
            <p className="text-base sm:text-lg text-gray-700 leading-relaxed">
              Our mission is to preserve, digitize, and make accessible historical and modern documents 
              written in Ranjana script using cutting-edge deep learning technology. The system recognizes 
              Ranjana characters from images and converts them to Devanagari (Nepali) script output, making 
              it easier to read, translate, and preserve these valuable texts.
            </p>
            <p className="text-base sm:text-lg text-gray-700 leading-relaxed">
              We combine the power of Convolutional Recurrent Neural Networks (CRNN) with modern web 
              technologies to create a Google Lens-style experience for Ranjana text recognition with 
              real-time AR visualization and translation support.
            </p>
          </div>
        </motion.div>

        {/* What We Do - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <h2 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-gray-800">What We Do</h2>
          <div className="space-y-4 sm:space-y-6">
            <div className="flex items-start space-x-3 sm:space-x-4">
              <div className="p-2 sm:p-3 rounded-xl bg-primary-100 flex-shrink-0">
                <FaSearch className="text-xl sm:text-2xl text-primary-600" />
              </div>
              <div>
                <h3 className="font-bold text-lg sm:text-xl mb-2 text-gray-800">Ranjana Character Recognition</h3>
                <p className="text-sm sm:text-base text-gray-700 leading-relaxed">
                  Our AI model recognizes individual Ranjana script characters from images with high accuracy, 
                  enabling precise text extraction and conversion to Devanagari script for easy reading and processing.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3 sm:space-x-4">
              <div className="p-2 sm:p-3 rounded-xl bg-green-100 flex-shrink-0">
                <FaEye className="text-xl sm:text-2xl text-green-600" />
              </div>
              <div>
                <h3 className="font-bold text-lg sm:text-xl mb-2 text-gray-800">AR Overlay Visualization</h3>
                <p className="text-sm sm:text-base text-gray-700 leading-relaxed">
                  Interactive augmented reality overlay shows recognized text with individual character bounding 
                  boxes, confidence scores, and hover tooltips - similar to Google Lens - making it easy to 
                  understand recognition results.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3 sm:space-x-4">
              <div className="p-2 sm:p-3 rounded-xl bg-orange-100 flex-shrink-0">
                <FaGlobe className="text-xl sm:text-2xl text-orange-600" />
              </div>
              <div>
                <h3 className="font-bold text-lg sm:text-xl mb-2 text-gray-800">Devanagari Output & Translation</h3>
                <p className="text-sm sm:text-base text-gray-700 leading-relaxed">
                  Automatically converts recognized Ranjana text to Devanagari (Nepali) characters. Optionally 
                  translate to English and other languages, helping bridge language barriers and make content 
                  accessible to a wider audience.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3 sm:space-x-4">
              <div className="p-2 sm:p-3 rounded-xl bg-purple-100 flex-shrink-0">
                <FaMobileAlt className="text-xl sm:text-2xl text-purple-600" />
              </div>
              <div>
                <h3 className="font-bold text-lg sm:text-xl mb-2 text-gray-800">Modern Web Interface</h3>
                <p className="text-sm sm:text-base text-gray-700 leading-relaxed">
                  Beautiful, fully responsive web application that works seamlessly on desktop, tablet, and mobile 
                  devices. Upload images or use your camera for real-time recognition, making OCR accessible to everyone.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Technology - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <h2 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-gray-800">Technology Stack</h2>
          <p className="text-base sm:text-lg text-gray-700 leading-relaxed mb-4">
            Lipika is built using a modern three-layer architecture for optimal performance and scalability:
          </p>
          <div className="space-y-4">
            <div className="flex items-start space-x-3 sm:space-x-4 p-3 sm:p-4 bg-blue-50 rounded-lg">
              <FaPython className="text-2xl sm:text-3xl text-blue-600 flex-shrink-0 mt-1" />
              <div>
                <h3 className="font-bold text-base sm:text-lg text-gray-800 mb-1">Model Layer (Python)</h3>
                <p className="text-sm sm:text-base text-gray-700">
                  PyTorch-based Improved Character CRNN neural network for character recognition from Ranjana script 
                  images. Trained for 500 epochs on comprehensive dataset. Flask REST API serves the model for 
                  real-time inference with character segmentation and recognition.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3 sm:space-x-4 p-3 sm:p-4 bg-orange-50 rounded-lg">
              <FaJava className="text-2xl sm:text-3xl text-orange-600 flex-shrink-0 mt-1" />
              <div>
                <h3 className="font-bold text-base sm:text-lg text-gray-800 mb-1">Presenter Layer (Java)</h3>
                <p className="text-sm sm:text-base text-gray-700">
                  Spring Boot microservices handle business logic, API orchestration, data validation, and admin 
                  dashboard. Manages OCR history, analytics, character statistics, and settings with in-memory 
                  storage for fast access.
                </p>
              </div>
            </div>
            <div className="flex items-start space-x-3 sm:space-x-4 p-3 sm:p-4 bg-cyan-50 rounded-lg">
              <FaReact className="text-2xl sm:text-3xl text-cyan-600 flex-shrink-0 mt-1" />
              <div>
                <h3 className="font-bold text-base sm:text-lg text-gray-800 mb-1">View Layer (React)</h3>
                <p className="text-sm sm:text-base text-gray-700">
                  Modern React 18 frontend with Vite, Tailwind CSS, and Framer Motion for intuitive user interface. 
                  Features responsive design, AR visualization, real-time OCR, and translation capabilities. 
                  Recharts for admin analytics dashboard.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Model Training - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <h2 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-gray-800 flex items-center">
            <FaChartLine className="mr-2 sm:mr-3 text-primary-600" />
            Model Training
          </h2>
          <p className="text-base sm:text-lg text-gray-700 leading-relaxed mb-4">
            Our CRNN model is trained on a comprehensive dataset with extensive data augmentation:
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
            <div className="bg-blue-50 p-3 sm:p-4 rounded-lg text-center">
              <div className="text-xl sm:text-2xl md:text-3xl font-bold text-blue-600 mb-1 sm:mb-2">500</div>
              <p className="text-xs sm:text-sm text-gray-700">Training Epochs</p>
            </div>
            <div className="bg-purple-50 p-3 sm:p-4 rounded-lg text-center">
              <div className="text-xl sm:text-2xl md:text-3xl font-bold text-purple-600 mb-1 sm:mb-2">62</div>
              <p className="text-xs sm:text-sm text-gray-700">Devanagari Characters</p>
            </div>
            <div className="bg-green-50 p-3 sm:p-4 rounded-lg text-center">
              <div className="text-xl sm:text-2xl md:text-3xl font-bold text-green-600 mb-1 sm:mb-2">80%</div>
              <p className="text-xs sm:text-sm text-gray-700">Training Split</p>
            </div>
            <div className="bg-pink-50 p-3 sm:p-4 rounded-lg text-center">
              <div className="text-xl sm:text-2xl md:text-3xl font-bold text-pink-600 mb-1 sm:mb-2">90%+</div>
              <p className="text-xs sm:text-sm text-gray-700">Accuracy</p>
            </div>
          </div>
          <div className="mt-4 sm:mt-6 p-3 sm:p-4 bg-gray-50 rounded-lg">
            <h4 className="font-semibold text-sm sm:text-base text-gray-800 mb-2">Training Features:</h4>
            <ul className="text-xs sm:text-sm text-gray-700 space-y-1 list-disc list-inside">
              <li>Data augmentation (rotation, affine, noise, blur, brightness, contrast)</li>
              <li>Dual learning rate schedulers (CosineAnnealingWarmRestarts & ReduceLROnPlateau)</li>
              <li>Periodic checkpoint saving every 5 epochs</li>
              <li>Automatic label conversion from English transliteration to Devanagari</li>
            </ul>
          </div>
        </motion.div>

        {/* Key Features - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <h2 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-gray-800">Key Features</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            {[
              { icon: '✓', text: 'Character-level recognition with individual bounding boxes' },
              { icon: '✓', text: 'Ranjana script input to Devanagari output conversion' },
              { icon: '✓', text: 'Confidence scoring for each recognized character' },
              { icon: '✓', text: 'Word grouping based on character spacing' },
              { icon: '✓', text: 'AR overlay visualization similar to Google Lens' },
              { icon: '✓', text: 'Optional English translation support' },
              { icon: '✓', text: 'Camera capture and image upload support' },
              { icon: '✓', text: 'Admin dashboard with analytics and OCR history' }
            ].map((item, index) => (
              <div key={index} className="flex items-start space-x-2 sm:space-x-3">
                <span className="text-green-600 text-lg sm:text-xl font-bold flex-shrink-0">{item.icon}</span>
                <span className="text-sm sm:text-base text-gray-700">{item.text}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Future Goals - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <h2 className="text-2xl sm:text-3xl font-bold mb-4 sm:mb-6 text-gray-800">Future Goals</h2>
          <ul className="space-y-3 text-sm sm:text-base text-gray-700">
            <li className="flex items-start">
              <span className="text-blue-600 mr-2 sm:mr-3 font-bold flex-shrink-0">→</span>
              <span>Improve model accuracy through additional training data and fine-tuning techniques</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2 sm:mr-3 font-bold flex-shrink-0">→</span>
              <span>Expand translation support to more languages including Hindi, Sanskrit, and regional languages</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2 sm:mr-3 font-bold flex-shrink-0">→</span>
              <span>Add text-to-speech functionality for accessibility and language learning</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2 sm:mr-3 font-bold flex-shrink-0">→</span>
              <span>Develop native mobile applications for iOS and Android platforms</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2 sm:mr-3 font-bold flex-shrink-0">→</span>
              <span>Support batch processing for large document archives and historical texts</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2 sm:mr-3 font-bold flex-shrink-0">→</span>
              <span>Integrate database storage for persistent OCR history and user management</span>
            </li>
          </ul>
        </motion.div>

        {/* Contact - Responsive */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="text-center card"
        >
          <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-800">Get Involved</h2>
          <p className="text-sm sm:text-base text-gray-700 mb-4 sm:mb-6">
            Lipika is an open-source project focused on preserving and digitizing Ranjana script. 
            Contributions, feedback, and suggestions are always welcome!
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-3 sm:gap-4">
            <motion.a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn-primary text-sm sm:text-base px-4 sm:px-6 py-2.5 sm:py-3"
            >
              View on GitHub
            </motion.a>
            <motion.a 
              href="mailto:contact@lipika.com" 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn-secondary text-sm sm:text-base px-4 sm:px-6 py-2.5 sm:py-3"
            >
              Contact Us
            </motion.a>
          </div>
        </motion.div>
      </main>
    </div>
  )
}

export default About
