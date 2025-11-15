import { motion } from 'framer-motion'
import { 
  FaCamera, FaRobot, FaEye, FaGlobe, FaBolt, FaBullseye, 
  FaChartBar, FaMobileAlt, FaLock, FaPython, FaCoffee, FaReact, FaUpload,
  FaShieldAlt, FaFont, FaArrowRight, FaStar, FaCheckCircle
} from 'react-icons/fa'

const Features = () => {
  const features = [
    {
      icon: FaUpload,
      title: 'Multiple Input Methods',
      description: 'Upload images from your device or capture in real-time using your camera. Supports JPG, PNG, WEBP, and BMP formats for maximum flexibility.',
      color: 'from-blue-500 to-cyan-500',
      bgColor: 'bg-blue-50'
    },
    {
      icon: FaRobot,
      title: 'Advanced AI Recognition',
      description: 'Powered by Character-based CRNN (Convolutional Recurrent Neural Network) deep learning model trained on 500 epochs for accurate character-level recognition of Ranjana script.',
      color: 'from-purple-500 to-pink-500',
      bgColor: 'bg-purple-50'
    },
    {
      icon: FaEye,
      title: 'Google Lens-Style AR Overlay',
      description: 'Interactive augmented reality visualization with individual character bounding boxes, confidence scores, and hover tooltips for detailed recognition insights.',
      color: 'from-green-500 to-emerald-500',
      bgColor: 'bg-green-50'
    },
    {
      icon: FaGlobe,
      title: 'Ranjana to Devanagari Output',
      description: 'Automatically converts recognized Ranjana script characters to Devanagari (Nepali) text output, with optional English translation support.',
      color: 'from-orange-500 to-red-500',
      bgColor: 'bg-orange-50'
    },
    {
      icon: FaBolt,
      title: 'Fast Processing',
      description: 'Optimized inference pipeline with GPU acceleration support. Process images with character segmentation and recognition in seconds.',
      color: 'from-indigo-500 to-purple-500',
      bgColor: 'bg-indigo-50'
    },
    {
      icon: FaBullseye,
      title: 'Character-Level Precision',
      description: 'Individual character detection with precise bounding boxes, confidence scores, and word grouping for perfect AR visualization.',
      color: 'from-pink-500 to-rose-500',
      bgColor: 'bg-pink-50'
    },
    {
      icon: FaChartBar,
      title: 'Confidence Scoring',
      description: 'Each recognized character includes a confidence score (0-100%) with color-coded visualization to help identify recognition accuracy.',
      color: 'from-teal-500 to-cyan-500',
      bgColor: 'bg-teal-50'
    },
    {
      icon: FaMobileAlt,
      title: 'Fully Responsive Design',
      description: 'Beautiful, modern UI that works seamlessly on desktop, tablet, and mobile devices with optimized layouts for every screen size.',
      color: 'from-yellow-500 to-orange-500',
      bgColor: 'bg-yellow-50'
    },
    {
      icon: FaLock,
      title: 'Secure & Private',
      description: 'All processing happens on your server. Images are processed in real-time and OCR history is stored securely with admin dashboard access.',
      color: 'from-gray-600 to-gray-800',
      bgColor: 'bg-gray-50'
    }
  ]

  const techStack = [
    { name: 'React 18 + Vite', description: 'Modern UI framework with fast build', icon: FaReact },
    { name: 'PyTorch', description: 'Deep learning framework for CRNN model', icon: FaRobot },
    { name: 'Flask', description: 'Python REST API for OCR service', icon: FaPython },
    { name: 'OpenCV', description: 'Advanced image processing and segmentation', icon: FaEye },
    { name: 'Tailwind CSS', description: 'Utility-first CSS framework', icon: FaFont },
    { name: 'Spring Boot', description: 'Java backend for API orchestration', icon: FaCoffee }
  ]

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.5,
        ease: "easeOut"
      }
    }
  }

  const cardHoverVariants = {
    rest: { scale: 1, y: 0 },
    hover: { 
      scale: 1.02, 
      y: -8,
      transition: {
        duration: 0.3,
        ease: "easeOut"
      }
    }
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary-600 via-purple-600 to-pink-600 pt-20 pb-16 sm:pt-24 sm:pb-20 md:pt-32 md:pb-24">
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.1, 0.2, 0.1],
              rotate: [0, 90, 0]
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute -top-40 -right-40 w-96 h-96 bg-white rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1, 1.3, 1],
              opacity: [0.1, 0.15, 0.1],
              rotate: [0, -90, 0]
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute -bottom-40 -left-40 w-96 h-96 bg-white rounded-full blur-3xl"
          />
        </div>

        <div className="relative z-10 container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="text-center"
          >
            {/* Icon with Animation */}
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200, damping: 15 }}
              className="flex justify-center items-center mb-6"
            >
              <div className="relative flex items-center justify-center">
                <div className="absolute inset-0 bg-white/20 rounded-full blur-xl"></div>
                <FaRobot className="relative text-6xl sm:text-7xl md:text-8xl text-white drop-shadow-2xl mx-auto" />
              </div>
            </motion.div>

            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-md rounded-full mb-6 border border-white/30"
            >
              <FaStar className="text-yellow-300 text-sm" />
              <span className="text-white text-sm font-semibold">Advanced OCR Technology</span>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold text-white mb-6 leading-tight"
            >
              Powerful Features for
              <br />
              <span className="bg-gradient-to-r from-yellow-300 via-pink-300 to-cyan-300 bg-clip-text text-transparent">
                Modern OCR
              </span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="text-lg sm:text-xl md:text-2xl text-white/90 max-w-3xl mx-auto mb-8 leading-relaxed"
            >
              Discover the cutting-edge capabilities of Lipika - Advanced Ranjana Script OCR System
              <br />
              <span className="text-base sm:text-lg text-white/80">
                Powered by deep learning and modern web technologies
              </span>
            </motion.p>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="flex flex-wrap justify-center gap-6 sm:gap-8 md:gap-12 mt-10"
            >
              {[
                { value: '90%+', label: 'Accuracy', icon: FaBullseye },
                { value: '<2s', label: 'Processing', icon: FaBolt },
                { value: '500', label: 'Epochs', icon: FaChartBar },
                { value: '62', label: 'Characters', icon: FaFont }
              ].map((stat, index) => {
                const StatIcon = stat.icon
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.7 + index * 0.1, type: "spring" }}
                    whileHover={{ scale: 1.1, y: -5 }}
                    className="text-center"
                  >
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <StatIcon className="text-white/80 text-xl" />
                      <div className="text-3xl sm:text-4xl font-bold text-white">{stat.value}</div>
                    </div>
                    <div className="text-sm sm:text-base text-white/80">{stat.label}</div>
                  </motion.div>
                )
              })}
            </motion.div>
          </motion.div>
        </div>

        {/* Wave Divider */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg className="w-full h-12 sm:h-16 md:h-20" viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 120L60 110C120 100 240 80 360 70C480 60 600 60 720 65C840 70 960 80 1080 85C1200 90 1320 90 1380 90L1440 90V120H1380C1320 120 1200 120 1080 120C960 120 840 120 720 120C600 120 480 120 360 120C240 120 120 120 60 120H0Z" fill="white"/>
          </svg>
        </div>
      </section>

      <main className="container mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-12 sm:py-16 md:py-20 max-w-7xl -mt-8 sm:-mt-12 md:-mt-16">

        {/* Features Grid - Responsive with Staggered Animation */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 md:gap-8 mb-8 sm:mb-12 md:mb-16"
        >
          {features.map((feature, index) => {
            const IconComponent = feature.icon
            return (
            <motion.div
              key={index}
              variants={itemVariants}
              initial="rest"
              whileHover="hover"
              variants={cardHoverVariants}
              className="card hover:shadow-2xl transition-all duration-300 relative overflow-hidden group"
            >
              {/* Background Gradient Effect on Hover */}
              <div className={`absolute inset-0 bg-gradient-to-r ${feature.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300`}></div>
              
              <div className="relative z-10">
                {/* Icon with Animated Background */}
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: index * 0.1 + 0.2, type: "spring", stiffness: 200 }}
                  className={`mb-3 sm:mb-4 inline-block p-3 sm:p-4 rounded-xl ${feature.bgColor} group-hover:scale-110 transition-transform duration-300`}
                >
                  <IconComponent className={`text-3xl sm:text-4xl md:text-5xl bg-gradient-to-r ${feature.color} bg-clip-text text-transparent`} />
                </motion.div>
                
                <motion.h3 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 + 0.3 }}
                  className="text-lg sm:text-xl font-bold mb-2 sm:mb-3 text-gray-800 group-hover:text-primary-600 transition-colors duration-300"
                >
                  {feature.title}
                </motion.h3>
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: index * 0.1 + 0.4 }}
                  className="text-sm sm:text-base text-gray-600 leading-relaxed"
                >
                  {feature.description}
                </motion.p>
              </div>
              
              {/* Decorative Arrow on Hover */}
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                whileHover={{ opacity: 1, x: 0 }}
                className="absolute top-4 right-4 text-primary-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
              >
                <FaArrowRight />
              </motion.div>
            </motion.div>
            )
          })}
        </motion.div>

        {/* Architecture Section - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="card bg-gradient-to-br from-gray-900 to-gray-800 text-white mb-8 sm:mb-12 md:mb-16 relative overflow-hidden"
        >
          {/* Animated Background Pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 left-0 w-64 h-64 bg-blue-500 rounded-full filter blur-3xl"></div>
            <div className="absolute bottom-0 right-0 w-64 h-64 bg-purple-500 rounded-full filter blur-3xl"></div>
          </div>
          
          <div className="relative z-10">
            <motion.h2 
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="text-2xl sm:text-3xl font-bold mb-6 sm:mb-8 text-center"
            >
              System Architecture
            </motion.h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8">
              {[
                { icon: FaPython, title: 'Model Layer', desc: 'Python-based CRNN model with Flask REST API for character recognition from Ranjana script images', color: 'text-yellow-400', delay: 0.1 },
                { icon: FaCoffee, title: 'Presenter Layer', desc: 'Java Spring Boot microservices for business logic, API orchestration, and admin dashboard', color: 'text-orange-400', delay: 0.2 },
                { icon: FaReact, title: 'View Layer', desc: 'React frontend with responsive design for intuitive user interface and AR visualization', color: 'text-cyan-400', delay: 0.3 }
              ].map((layer, index) => {
                const LayerIcon = layer.icon
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: layer.delay }}
                    whileHover={{ scale: 1.05, y: -5 }}
                    className="text-center p-4 sm:p-6 bg-white/5 rounded-lg backdrop-blur-sm hover:bg-white/10 transition-all duration-300"
                  >
                    <motion.div
                      animate={{ rotate: [0, 10, -10, 0] }}
                      transition={{ repeat: Infinity, duration: 3, delay: index * 0.5 }}
                      className="text-3xl sm:text-4xl mb-3 sm:mb-4 flex justify-center"
                    >
                      <LayerIcon className={layer.color} />
                    </motion.div>
                    <h3 className="text-lg sm:text-xl font-bold mb-2">{layer.title}</h3>
                    <p className="text-sm sm:text-base text-gray-300">{layer.desc}</p>
                  </motion.div>
                )
              })}
            </div>
          </div>
        </motion.div>

        {/* Technology Stack - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="card mb-8 sm:mb-12 md:mb-16"
        >
          <motion.h2 
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="text-2xl sm:text-3xl font-bold mb-6 sm:mb-8 text-center text-gray-800"
          >
            Technology Stack
          </motion.h2>
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4 md:gap-6"
          >
            {techStack.map((tech, index) => {
              const TechIcon = tech.icon
              return (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  whileHover={{ scale: 1.05, y: -5 }}
                  className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-all duration-300 group cursor-pointer"
                >
                  <motion.div
                    animate={{ rotate: [0, 5, -5, 0] }}
                    transition={{ repeat: Infinity, duration: 4, delay: index * 0.3 }}
                    className="mb-2 sm:mb-3 flex justify-center"
                  >
                    <TechIcon className="text-2xl sm:text-3xl text-primary-600 group-hover:scale-110 transition-transform duration-300" />
                  </motion.div>
                  <h3 className="font-bold text-sm sm:text-base md:text-lg text-gray-800 mb-1">{tech.name}</h3>
                  <p className="text-xs sm:text-sm text-gray-600">{tech.description}</p>
                </motion.div>
              )
            })}
          </motion.div>
        </motion.div>

        {/* Performance Metrics - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 md:gap-6"
        >
          {[
            { value: '<2s', label: 'Processing Time', color: 'text-primary-600', icon: FaBolt },
            { value: '62', label: 'Devanagari Characters', color: 'text-green-600', icon: FaFont },
            { value: '500', label: 'Training Epochs', color: 'text-purple-600', icon: FaChartBar },
            { value: '90%+', label: 'Model Accuracy', color: 'text-pink-600', icon: FaBullseye }
          ].map((metric, index) => {
            const MetricIcon = metric.icon
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1, type: "spring", stiffness: 200 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="card text-center group cursor-pointer"
              >
                <motion.div
                  animate={{ rotate: [0, 10, -10, 0] }}
                  transition={{ repeat: Infinity, duration: 2, delay: index * 0.3 }}
                  className="mb-2 flex justify-center"
                >
                  <MetricIcon className={`text-2xl sm:text-3xl ${metric.color} group-hover:scale-110 transition-transform duration-300`} />
                </motion.div>
                <div className={`text-2xl sm:text-3xl md:text-4xl font-bold mb-1 sm:mb-2 ${metric.color}`}>
                  {metric.value}
                </div>
                <p className="text-xs sm:text-sm text-gray-600">{metric.label}</p>
              </motion.div>
            )
          })}
        </motion.div>
      </main>
    </div>
  )
}

export default Features
