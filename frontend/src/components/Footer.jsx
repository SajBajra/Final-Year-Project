import { motion } from 'framer-motion'
import { FaScroll, FaHeart } from 'react-icons/fa'

const Footer = () => {
  return (
    <footer className="relative mt-20 bg-primary-800 text-white overflow-hidden">
      {/* Decorative Elements */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-0 left-0 w-96 h-96 bg-blue-500 rounded-full filter blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-purple-500 rounded-full filter blur-3xl"></div>
      </div>
      
      <div className="container mx-auto px-4 py-12 max-w-6xl relative z-10">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {/* Brand Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            <div className="flex items-center space-x-3 mb-4">
              <FaScroll className="text-4xl text-secondary-400" />
              <h3 className="text-3xl font-black text-white">
                Lipika
              </h3>
            </div>
            <p className="text-gray-300 text-sm leading-relaxed mb-4">
              Advanced OCR system for Ranjana script with AR support, powered by cutting-edge deep learning technology.
            </p>
            <div className="flex space-x-4">
              <motion.a
                whileHover={{ scale: 1.2, y: -2 }}
                whileTap={{ scale: 0.9 }}
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="w-10 h-10 bg-white/10 hover:bg-white/20 backdrop-blur-sm rounded-lg flex items-center justify-center transition-all duration-300"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </motion.a>
            </div>
          </motion.div>
          
          {/* Technology Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <h4 className="font-bold text-lg mb-4 text-white">
              Technology
            </h4>
            <ul className="space-y-2 text-sm text-gray-300">
              {[
                'Python + PyTorch',
                'React + TailwindCSS',
                'CRNN Architecture',
                'AR Visualization',
                'Flask REST API',
                'Deep Learning'
              ].map((tech, idx) => (
                <motion.li
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex items-center space-x-2 hover:text-white transition-colors"
                >
                  <span className="text-secondary-400">▸</span>
                  <span>{tech}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>
          
          {/* Features Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <h4 className="font-bold text-lg mb-4 text-white">
              Features
            </h4>
            <ul className="space-y-2 text-sm text-gray-300">
              {[
                'Character Recognition',
                'Google Lens Style AR',
                'Real-time Processing',
                'Camera Capture',
                'Multi-language Support',
                'High Accuracy'
              ].map((feature, idx) => (
                <motion.li
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: idx * 0.05 }}
                  className="flex items-center space-x-2 hover:text-white transition-colors"
                >
                  <span className="text-secondary-400">▸</span>
                  <span>{feature}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>
        </div>
        
        {/* Bottom Bar */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="border-t border-white/10 pt-6 mt-8"
        >
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <p className="text-sm text-gray-400 text-center md:text-left">
              © 2024 <span className="text-white font-semibold">Lipika Project</span>. 
              Built for Ranjana script preservation and digitization.
            </p>
            <div className="flex items-center space-x-2 text-xs text-gray-400">
              <span>Made By</span>
              <span className="text-white font-semibold">Sajesh Bajracharya</span>
            </div>
          </div>
        </motion.div>
      </div>
    </footer>
  )
}

export default Footer

