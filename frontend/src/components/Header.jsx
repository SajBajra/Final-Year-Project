import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { FaScroll } from 'react-icons/fa'

const Header = () => {
  const location = useLocation()

  const isActive = (path) => location.pathname === path

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 border-b border-white/20 shadow-sm">
      <div className="container mx-auto px-4 py-4 max-w-7xl">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-3 group">
            <motion.div
              whileHover={{ rotate: [0, -10, 10, -10, 0] }}
              transition={{ duration: 0.5 }}
              className="text-4xl text-primary-600"
            >
              <FaScroll />
            </motion.div>
            <div>
              <motion.h1
                whileHover={{ scale: 1.05 }}
                className="text-2xl font-black bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent"
              >
                Lipika
              </motion.h1>
              <p className="text-xs text-gray-500 group-hover:text-gray-700 transition-colors">
                Ranjana OCR System
              </p>
            </div>
          </Link>
          
          <nav className="hidden md:flex items-center space-x-1">
            {[
              { path: '/', label: 'Home' },
              { path: '/features', label: 'Features' },
              { path: '/about', label: 'About' }
            ].map((item, index) => (
              <motion.div
                key={item.path}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Link 
                  to={item.path}
                  className="relative px-4 py-2 rounded-lg font-medium transition-all duration-300 group"
                >
                  <span className={`relative z-10 ${
                    isActive(item.path)
                      ? 'text-blue-600 font-semibold'
                      : 'text-gray-600 group-hover:text-blue-600'
                  }`}>
                    {item.label}
                  </span>
                  
                  {isActive(item.path) && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute inset-0 bg-gradient-to-r from-blue-100 to-purple-100 rounded-lg opacity-50"
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    />
                  )}
                  
                  <div className="absolute inset-0 bg-gray-100 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"></div>
                </Link>
              </motion.div>
            ))}
            
            <motion.a
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="ml-4 p-2 bg-gray-100 hover:bg-gradient-to-r hover:from-blue-500 hover:to-purple-500 rounded-lg text-gray-600 hover:text-white transition-all duration-300 shadow-md hover:shadow-lg"
              title="GitHub"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </motion.a>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header

