import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Logo from '../images/HeroSection-FrontPage.png'

const Preloader = () => {
  const [loading, setLoading] = useState(true)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    // Simulate loading progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setTimeout(() => setLoading(false), 300)
          return 100
        }
        return prev + 2
      })
    }, 30)

    // Ensure preloader shows for at least 1.5 seconds
    const minLoadTime = setTimeout(() => {
      if (progress >= 100) {
        setLoading(false)
      }
    }, 1500)

    return () => {
      clearInterval(interval)
      clearTimeout(minLoadTime)
    }
  }, [])

  return (
    <AnimatePresence>
      {loading && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 1.1 }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
          className="fixed inset-0 z-[9999] bg-white flex items-center justify-center"
        >

          <div className="relative z-10 flex flex-col items-center">
            {/* Logo container with reveal animation */}
            <div className="relative mb-8" style={{ width: '200px', height: '350px', maxHeight: '50vh' }}>
              {/* Logo with clip-path reveal from top to bottom */}
              <motion.div
                initial={{ clipPath: 'inset(0 0 100% 0)' }}
                animate={{ clipPath: 'inset(0 0 0% 0)' }}
                transition={{
                  duration: 1.5,
                  ease: [0.43, 0.13, 0.23, 0.96]
                }}
                className="absolute inset-0 flex items-center justify-center"
              >
                <img
                  src={Logo}
                  alt="Logo"
                  className="w-full h-full object-contain drop-shadow-xl"
                  style={{
                    imageRendering: 'crisp-edges',
                    imageRendering: '-webkit-optimize-contrast',
                    transform: 'translateZ(0)',
                    backfaceVisibility: 'hidden',
                    WebkitFontSmoothing: 'subpixel-antialiased'
                  }}
                  loading="eager"
                  decoding="sync"
                />
              </motion.div>

              {/* Animated glow effect */}
              <motion.div
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.1, 0.3, 0.1]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="absolute inset-0 bg-primary-600/30 rounded-full blur-3xl"
              />
            </div>

            {/* Loading text with fade animation */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.5 }}
              className="text-center"
            >
              <h2 className="text-2xl md:text-3xl font-bold text-primary-700 mb-4 tracking-wide">
                Lipika
              </h2>
              <p className="text-primary-600/70 text-sm md:text-base mb-6">
                Ranjana Script OCR
              </p>

              {/* Progress bar */}
              <div className="w-64 md:w-80 h-2 bg-gray-200 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: '0%' }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.1 }}
                  className="h-full bg-gradient-to-r from-primary-500 to-primary-600 rounded-full shadow-lg"
                />
              </div>

              {/* Progress percentage */}
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="text-primary-600/60 text-xs mt-3 font-medium"
              >
                {progress}%
              </motion.p>
            </motion.div>

            {/* Animated dots */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
              className="flex gap-2 mt-6"
            >
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  animate={{
                    y: [0, -10, 0],
                  }}
                  transition={{
                    duration: 0.8,
                    repeat: Infinity,
                    delay: i * 0.2,
                    ease: "easeInOut"
                  }}
                  className="w-2 h-2 bg-primary-600 rounded-full"
                />
              ))}
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default Preloader

