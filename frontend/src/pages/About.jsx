import { motion } from 'framer-motion'
import { 
  FaSearch, FaEye, FaGlobe, FaMobileAlt, FaPython, FaJava, FaReact, 
  FaDatabase, FaChartLine, FaChartBar, FaScroll, FaRobot, FaCode, FaServer,
  FaArrowRight, FaCheckCircle, FaLightbulb, FaShieldAlt, FaStar, FaHeart
} from 'react-icons/fa'
import aboutSectionImage from '../images/AboutSection_1.png'

const About = () => {
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
      y: -5,
      transition: {
        duration: 0.3,
        ease: "easeOut"
      }
    }
  }

  return (
    <div className="min-h-screen bg-primary-50">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-primary-600 flex items-center" style={{ minHeight: '90vh' }}>
        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <motion.div
            animate={{
              scale: [1, 1.3, 1],
              opacity: [0.1, 0.2, 0.1],
              rotate: [0, 120, 0]
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute top-20 -right-20 w-96 h-96 bg-white rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.1, 0.15, 0.1],
              rotate: [0, -120, 0]
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute bottom-20 -left-20 w-96 h-96 bg-white rounded-full blur-3xl"
          />
          <motion.div
            animate={{
              scale: [1, 1.4, 1],
              opacity: [0.05, 0.1, 0.05],
              x: [0, 100, 0],
              y: [0, 50, 0]
            }}
            transition={{
              duration: 30,
              repeat: Infinity,
              ease: "linear"
            }}
            className="absolute top-1/2 left-1/2 w-80 h-80 bg-white rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2"
          />
        </div>

        <div className="relative z-10 container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="text-center"
          >
            {/* Hero Image with Animation */}
            <motion.div
              initial={{ scale: 0, rotate: -10 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200, damping: 15 }}
              className="flex justify-center items-center mb-6"
            >
              <div className="relative flex items-center justify-center">
                <div className="absolute inset-0 bg-white/20 rounded-3xl blur-2xl"></div>
                <img 
                  src={aboutSectionImage} 
                  alt="About Lipika OCR" 
                  className="relative h-32 sm:h-40 md:h-48 lg:h-56 w-auto drop-shadow-2xl mx-auto"
                />
              </div>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold text-white mb-6 leading-tight"
            >
              About Lipika
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="text-lg sm:text-xl md:text-2xl text-white/90 max-w-3xl mx-auto mb-8 leading-relaxed"
            >
              Preserving and digitizing Ranjana script through
              <br />
              <span className="text-base sm:text-lg text-white/80">
                advanced AI technology and modern web development
              </span>
            </motion.p>

            {/* Key Points */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="flex flex-wrap justify-center gap-4 sm:gap-6 md:gap-8 mt-10"
            >
              {[
                { icon: FaRobot, text: 'AI-Powered' },
                { icon: FaGlobe, text: 'Cultural Preservation' },
                { icon: FaStar, text: 'Cutting-Edge Tech' }
              ].map((point, index) => {
                const PointIcon = point.icon
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.7 + index * 0.1, type: "spring" }}
                    whileHover={{ scale: 1.1, y: -5 }}
                    className="flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-md rounded-full border border-white/20"
                  >
                    <PointIcon className="text-white text-lg" />
                    <span className="text-white text-sm sm:text-base font-medium">{point.text}</span>
                  </motion.div>
                )
              })}
            </motion.div>
          </motion.div>
        </div>
      </section>

      <main className="container mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-4 sm:py-6 md:py-8 max-w-7xl">

        {/* Mission Section - Split into multiple cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 mb-8 sm:mb-12">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6 }}
            className="card relative overflow-hidden group"
          >
            <div className="absolute inset-0 bg-primary-50 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative z-10">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 rounded-xl bg-primary-100">
                  <FaLightbulb className="text-2xl text-primary-600" />
                </div>
                <h2 className="text-xl font-bold text-gray-800">What is Lipika?</h2>
              </div>
              <p className="text-base text-gray-700 leading-relaxed">
                <strong>Lipika</strong> (लिपिका) is an advanced Optical Character Recognition (OCR) system 
                designed specifically for the Ranjana script, an ancient and beautiful writing system used 
                in Nepal, Tibet, and other Himalayan regions.
              </p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="card relative overflow-hidden group"
          >
            <div className="absolute inset-0 bg-primary-50 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative z-10">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 rounded-xl bg-primary-100">
                  <FaRobot className="text-2xl text-primary-600" />
                </div>
                <h2 className="text-xl font-bold text-gray-800">Our Mission</h2>
              </div>
              <p className="text-base text-gray-700 leading-relaxed">
                To preserve, digitize, and make accessible historical and modern documents 
                written in Ranjana script using cutting-edge deep learning technology.
              </p>
            </div>
          </motion.div>
        </div>

        {/* Centered Heading Section */}
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8 sm:mb-12"
        >
        
        </motion.div>

        {/* What We Do Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6 }}
          className="mb-8 sm:mb-12"
        >
          <div className="text-center mb-6">
            <div className="flex items-center justify-center space-x-3 mb-3">
              <div className="p-2 rounded-xl bg-green-100">
                <FaRobot className="text-2xl text-primary-600" />
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">What We Do</h2>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
            {[
              {
                icon: FaSearch,
                title: 'Ranjana Character Recognition',
                description: 'Our AI model recognizes individual Ranjana script characters from images with high accuracy, enabling precise text extraction and conversion to Devanagari script.',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600',
                delay: 0.1
              },
              {
                icon: FaEye,
                title: 'AR Overlay Visualization',
                description: 'Interactive augmented reality overlay shows recognized text with individual character bounding boxes, confidence scores, and hover tooltips.',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600',
                delay: 0.2
              },
              {
                icon: FaGlobe,
                title: 'Devanagari Output & Translation',
                description: 'Automatically converts recognized Ranjana text to Devanagari (Nepali) characters. Optionally translate to English and other languages.',
                bgColor: 'bg-secondary-100',
                iconColor: 'text-secondary-600',
                delay: 0.3
              },
              {
                icon: FaMobileAlt,
                title: 'Modern Web Interface',
                description: 'Beautiful, fully responsive web application that works seamlessly on desktop, tablet, and mobile devices. Upload images or use your camera for real-time recognition.',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600',
                delay: 0.4
              }
            ].map((item, index) => {
              const ItemIcon = item.icon
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: item.delay }}
                  whileHover={{ scale: 1.02, y: -5 }}
                  className="card group"
                >
                  <div className="flex items-start space-x-4">
                    <div className={`p-3 rounded-xl ${item.bgColor} flex-shrink-0 group-hover:shadow-lg transition-shadow duration-300`}>
                      <ItemIcon className={`text-2xl ${item.iconColor}`} />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-bold text-lg mb-2 text-gray-800 group-hover:text-primary-600 transition-colors duration-300">
                        {item.title}
                      </h3>
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {item.description}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.div>

        {/* Technology Stack Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mb-8 sm:mb-12"
        >
          <div className="text-center mb-6">
            <div className="flex items-center justify-center space-x-3 mb-3">
              <div className="p-2 rounded-xl bg-blue-100">
                <FaCode className="text-2xl text-blue-600" />
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">Technology Stack</h2>
            </div>
            <p className="text-base text-gray-700 leading-relaxed max-w-3xl mx-auto">
              Lipika is built using a modern three-layer architecture for optimal performance and scalability:
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
            {[
              {
                icon: FaPython,
                title: 'Model Layer',
                subtitle: 'Python',
                description: 'PyTorch-based Improved Character CRNN neural network for character recognition from Ranjana script images. Trained for 500 epochs on comprehensive dataset. Flask REST API serves the model for real-time inference with character segmentation and recognition.',
                bgColor: 'bg-blue-50',
                iconColor: 'text-blue-600',
                borderColor: 'border-blue-200',
                delay: 0.1
              },
              {
                icon: FaJava,
                title: 'Presenter Layer',
                subtitle: 'Java',
                description: 'Spring Boot microservices handle business logic, API orchestration, data validation, and admin dashboard. Manages OCR history, analytics, character statistics, and settings with MySQL database for persistent storage.',
                bgColor: 'bg-orange-50',
                iconColor: 'text-secondary-600',
                borderColor: 'border-orange-200',
                delay: 0.2
              },
              {
                icon: FaReact,
                title: 'View Layer',
                subtitle: 'React',
                description: 'Modern React 18 frontend with Vite, Tailwind CSS, and Framer Motion for intuitive user interface. Features responsive design, AR visualization, real-time OCR, and translation capabilities. Recharts for admin analytics dashboard.',
                bgColor: 'bg-cyan-50',
                iconColor: 'text-cyan-600',
                borderColor: 'border-cyan-200',
                delay: 0.3
              }
            ].map((layer, index) => {
              const LayerIcon = layer.icon
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: layer.delay }}
                  whileHover={{ scale: 1.02, y: -5 }}
                  className={`card border-2 ${layer.borderColor} group`}
                >
                  <div className={`p-4 rounded-xl ${layer.bgColor} mb-4 group-hover:shadow-lg transition-shadow duration-300 inline-block`}>
                    <LayerIcon className={`text-3xl ${layer.iconColor}`} />
                  </div>
                  <div>
                    <h3 className="font-bold text-lg text-gray-800 mb-1 group-hover:text-primary-600 transition-colors duration-300">
                      {layer.title}
                    </h3>
                    <p className="text-sm text-gray-600 mb-3 font-medium">
                      {layer.subtitle}
                    </p>
                    <p className="text-sm text-gray-700 leading-relaxed">
                      {layer.description}
                    </p>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.div>

        {/* Model Training - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6"
          >
            <div className="p-2 sm:p-3 rounded-xl bg-purple-100">
              <FaChartLine className="text-2xl sm:text-3xl text-purple-600" />
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">Model Training</h2>
          </motion.div>
          <p className="text-base sm:text-lg text-gray-700 leading-relaxed mb-4">
            Our CRNN model is trained on a comprehensive dataset with extensive data augmentation:
          </p>
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mb-4 sm:mb-6"
          >
            {[
              { value: '500', label: 'Training Epochs', color: 'text-blue-600', bgColor: 'bg-blue-50' },
              { value: '62', label: 'Devanagari Characters', color: 'text-purple-600', bgColor: 'bg-purple-50' },
              { value: '80%', label: 'Training Split', color: 'text-black', bgColor: 'bg-secondary-50' },
              { value: '90%+', label: 'Accuracy', color: 'text-pink-600', bgColor: 'bg-pink-50' }
            ].map((metric, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ scale: 1.05, y: -5 }}
                className={`${metric.bgColor} p-3 sm:p-4 rounded-lg text-center group cursor-pointer`}
              >
                <motion.div
                  transition={{ repeat: Infinity, duration: 2, delay: index * 0.2 }}
                  className={`text-xl sm:text-2xl md:text-3xl font-bold mb-1 sm:mb-2 ${metric.color}`}
                >
                  {metric.value}
                </motion.div>
                <p className="text-xs sm:text-sm text-gray-700">{metric.label}</p>
              </motion.div>
            ))}
          </motion.div>
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
            className="p-3 sm:p-4 bg-gray-50 rounded-lg"
          >
            <h4 className="font-semibold text-sm sm:text-base text-gray-800 mb-2 sm:mb-3 flex items-center">
              <FaCheckCircle className="text-green-600 mr-2" />
              Training Features:
            </h4>
            <motion.ul
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="text-xs sm:text-sm text-gray-700 space-y-1 sm:space-y-2 list-disc list-inside"
            >
              {[
                'Data augmentation (rotation, affine, noise, blur, brightness, contrast)',
                'Dual learning rate schedulers (CosineAnnealingWarmRestarts & ReduceLROnPlateau)',
                'Periodic checkpoint saving every 5 epochs',
                'Automatic label conversion from English transliteration to Devanagari'
              ].map((feature, index) => (
                <motion.li key={index} variants={itemVariants} className="flex items-start">
                  <span className="text-primary-600 mr-2">•</span>
                  <span>{feature}</span>
                </motion.li>
              ))}
            </motion.ul>
          </motion.div>
        </motion.div>

        {/* Key Features - Enhanced Layout with Icons */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mb-8 sm:mb-12"
        >
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="text-center mb-6 sm:mb-8"
          >
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="p-2 rounded-xl bg-primary-100">
                <FaCheckCircle className="text-3xl text-primary-600" />
              </div>
              <h2 className="text-3xl sm:text-4xl font-bold text-primary-600">
                Key Features
              </h2>
            </div>
            <p className="text-black text-center max-w-2xl mx-auto">
              Discover the powerful capabilities that make Lipika the premier OCR solution for Ranjana script
            </p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6"
          >
            {[
              {
                icon: FaSearch,
                title: 'Character Recognition',
                description: 'Individual bounding boxes with precise character-level recognition',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600'
              },
              {
                icon: FaGlobe,
                title: 'Script Conversion',
                description: 'Ranjana to Devanagari output conversion seamlessly',
                bgColor: 'bg-secondary-100',
                iconColor: 'text-secondary-600'
              },
              {
                icon: FaChartLine,
                title: 'Confidence Scoring',
                description: 'Real-time confidence scores for each recognized character',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600'
              },
              {
                icon: FaEye,
                title: 'AR Visualization',
                description: 'AR overlay with interactive bounding boxes',
                bgColor: 'bg-secondary-100',
                iconColor: 'text-secondary-600'
              },
              {
                icon: FaMobileAlt,
                title: 'Multi-Input Support',
                description: 'Camera capture and image upload for flexible usage',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600'
              },
              {
                icon: FaGlobe,
                title: 'Translation Support',
                description: 'Optional English translation to bridge language barriers',
                bgColor: 'bg-secondary-100',
                iconColor: 'text-secondary-600'
              },
              {
                icon: FaDatabase,
                title: 'Word Grouping',
                description: 'Intelligent word grouping based on character spacing',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600'
              },
              {
                icon: FaChartBar,
                title: 'Admin Dashboard',
                description: 'Comprehensive analytics and OCR history management',
                bgColor: 'bg-secondary-100',
                iconColor: 'text-secondary-600'
              }
            ].map((feature, index) => {
              const FeatureIcon = feature.icon
              return (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="card group relative overflow-hidden cursor-pointer"
                >
                  {/* Gradient border effect */}
                  <div className="absolute inset-0 bg-primary-50 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  
                  <div className="relative z-10">
                    <div className={`w-14 h-14 ${feature.bgColor} rounded-xl flex items-center justify-center mb-4`}>
                      <FeatureIcon className={`text-2xl ${feature.iconColor}`} />
                    </div>
                    <h3 className="font-bold text-lg mb-2 text-gray-800 group-hover:text-primary-600 transition-colors duration-300">
                      {feature.title}
                    </h3>
                    <p className="text-sm text-gray-600 leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              )
            })}
          </motion.div>
        </motion.div>

        {/* Future Goals - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6"
          >
            <div className="p-2 sm:p-3 rounded-xl bg-yellow-100">
              <FaLightbulb className="text-2xl sm:text-3xl text-yellow-600" />
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">Future Goals</h2>
          </motion.div>
          <motion.ul
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="space-y-3 text-sm sm:text-base text-gray-700"
          >
            {[
              'Improve model accuracy through additional training data and fine-tuning techniques',
              'Add text-to-speech functionality for accessibility and language learning',
              'Develop native mobile applications for iOS and Android platforms',
              'Support batch processing for large document archives and historical texts'
            ].map((goal, index) => (
              <motion.li
                key={index}
                variants={itemVariants}
                className="flex items-start space-x-2 sm:space-x-3 p-2 sm:p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-300 group"
              >
                <motion.span
                  animate={{ x: [0, 5, 0] }}
                  transition={{ repeat: Infinity, duration: 2, delay: index * 0.3 }}
                  className="text-primary-600 text-lg sm:text-xl font-bold flex-shrink-0"
                >
                  →
                </motion.span>
                <span className="group-hover:text-gray-900 transition-colors duration-300">
                  {goal}
                </span>
              </motion.li>
            ))}
          </motion.ul>
        </motion.div>

      </main>
    </div>
  )
}

export default About
