import { motion } from 'framer-motion'
import { 
  FaSearch, FaEye, FaGlobe, FaMobileAlt, FaPython, FaJava, FaReact, 
  FaDatabase, FaChartLine, FaScroll, FaRobot, FaCode, FaServer,
  FaArrowRight, FaCheckCircle, FaLightbulb, FaShieldAlt, FaStar, FaHeart
} from 'react-icons/fa'

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
    <div className="min-h-screen bg-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 pt-20 pb-16 sm:pt-24 sm:pb-20 md:pt-32 md:pb-24">
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
            {/* Icon with Animation */}
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200, damping: 15 }}
              className="inline-block mb-6"
            >
              <div className="relative">
                <div className="absolute inset-0 bg-white/20 rounded-full blur-xl"></div>
                <FaScroll className="relative text-6xl sm:text-7xl md:text-8xl text-white drop-shadow-2xl" />
              </div>
            </motion.div>

            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-md rounded-full mb-6 border border-white/30"
            >
              <FaHeart className="text-pink-300 text-sm" />
              <span className="text-white text-sm font-semibold">Preserving Cultural Heritage</span>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold text-white mb-6 leading-tight"
            >
              About
              <br />
              <span className="bg-gradient-to-r from-yellow-300 via-pink-300 to-cyan-300 bg-clip-text text-transparent">
                Lipika
              </span>
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
                { icon: FaCode, text: 'Open Source' },
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

        {/* Wave Divider */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg className="w-full h-12 sm:h-16 md:h-20" viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 120L60 110C120 100 240 80 360 70C480 60 600 60 720 65C840 70 960 80 1080 85C1200 90 1320 90 1380 90L1440 90V120H1380C1320 120 1200 120 1080 120C960 120 840 120 720 120C600 120 480 120 360 120C240 120 120 120 60 120H0Z" fill="white"/>
          </svg>
        </div>
      </section>

      <main className="container mx-auto px-3 sm:px-4 md:px-6 lg:px-8 py-12 sm:py-16 md:py-20 max-w-4xl -mt-8 sm:-mt-12 md:-mt-16">

        {/* Mission Section - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6 }}
          className="card mb-6 sm:mb-8 md:mb-12 relative overflow-hidden group"
        >
          {/* Decorative background */}
          <div className="absolute inset-0 bg-gradient-to-r from-primary-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          
          <div className="relative z-10">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6"
            >
              <div className="p-2 sm:p-3 rounded-xl bg-primary-100">
                <FaLightbulb className="text-2xl sm:text-3xl text-primary-600" />
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">Our Mission</h2>
            </motion.div>
            <motion.div
              variants={containerVariants}
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              className="space-y-4"
            >
              <motion.p variants={itemVariants} className="text-base sm:text-lg text-gray-700 leading-relaxed">
                <strong>Lipika</strong> (लिपिका) is an advanced Optical Character Recognition (OCR) system 
                designed specifically for the Ranjana script, an ancient and beautiful writing system used 
                in Nepal, Tibet, and other Himalayan regions.
              </motion.p>
              <motion.p variants={itemVariants} className="text-base sm:text-lg text-gray-700 leading-relaxed">
                Our mission is to preserve, digitize, and make accessible historical and modern documents 
                written in Ranjana script using cutting-edge deep learning technology. The system recognizes 
                Ranjana characters from images and converts them to Devanagari (Nepali) script output, making 
                it easier to read, translate, and preserve these valuable texts.
              </motion.p>
              <motion.p variants={itemVariants} className="text-base sm:text-lg text-gray-700 leading-relaxed">
                We combine the power of Convolutional Recurrent Neural Networks (CRNN) with modern web 
                technologies to create a Google Lens-style experience for Ranjana text recognition with 
                real-time AR visualization and translation support.
              </motion.p>
            </motion.div>
          </div>
        </motion.div>

        {/* What We Do - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6"
          >
            <div className="p-2 sm:p-3 rounded-xl bg-green-100">
              <FaRobot className="text-2xl sm:text-3xl text-green-600" />
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">What We Do</h2>
          </motion.div>
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="space-y-4 sm:space-y-6"
          >
            {[
              {
                icon: FaSearch,
                title: 'Ranjana Character Recognition',
                description: 'Our AI model recognizes individual Ranjana script characters from images with high accuracy, enabling precise text extraction and conversion to Devanagari script for easy reading and processing.',
                bgColor: 'bg-primary-100',
                iconColor: 'text-primary-600',
                delay: 0.1
              },
              {
                icon: FaEye,
                title: 'AR Overlay Visualization',
                description: 'Interactive augmented reality overlay shows recognized text with individual character bounding boxes, confidence scores, and hover tooltips - similar to Google Lens - making it easy to understand recognition results.',
                bgColor: 'bg-green-100',
                iconColor: 'text-green-600',
                delay: 0.2
              },
              {
                icon: FaGlobe,
                title: 'Devanagari Output & Translation',
                description: 'Automatically converts recognized Ranjana text to Devanagari (Nepali) characters. Optionally translate to English and other languages, helping bridge language barriers and make content accessible to a wider audience.',
                bgColor: 'bg-orange-100',
                iconColor: 'text-orange-600',
                delay: 0.3
              },
              {
                icon: FaMobileAlt,
                title: 'Modern Web Interface',
                description: 'Beautiful, fully responsive web application that works seamlessly on desktop, tablet, and mobile devices. Upload images or use your camera for real-time recognition, making OCR accessible to everyone.',
                bgColor: 'bg-purple-100',
                iconColor: 'text-purple-600',
                delay: 0.4
              }
            ].map((item, index) => {
              const ItemIcon = item.icon
              return (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  initial="rest"
                  whileHover="hover"
                  variants={cardHoverVariants}
                  className="flex items-start space-x-3 sm:space-x-4 p-4 sm:p-5 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-300 group"
                >
                  <motion.div
                    whileHover={{ scale: 1.1, rotate: 5 }}
                    className={`p-2 sm:p-3 rounded-xl ${item.bgColor} flex-shrink-0 group-hover:shadow-lg transition-shadow duration-300`}
                  >
                    <ItemIcon className={`text-xl sm:text-2xl ${item.iconColor}`} />
                  </motion.div>
                  <div className="flex-1">
                    <motion.h3
                      whileHover={{ x: 5 }}
                      className="font-bold text-lg sm:text-xl mb-2 text-gray-800 group-hover:text-primary-600 transition-colors duration-300"
                    >
                      {item.title}
                    </motion.h3>
                    <p className="text-sm sm:text-base text-gray-700 leading-relaxed">
                      {item.description}
                    </p>
                  </div>
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    whileHover={{ opacity: 1, x: 0 }}
                    className="text-primary-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex-shrink-0"
                  >
                    <FaArrowRight />
                  </motion.div>
                </motion.div>
              )
            })}
          </motion.div>
        </motion.div>

        {/* Technology - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6"
          >
            <div className="p-2 sm:p-3 rounded-xl bg-blue-100">
              <FaCode className="text-2xl sm:text-3xl text-blue-600" />
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">Technology Stack</h2>
          </motion.div>
          <p className="text-base sm:text-lg text-gray-700 leading-relaxed mb-4">
            Lipika is built using a modern three-layer architecture for optimal performance and scalability:
          </p>
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="space-y-4"
          >
            {[
              {
                icon: FaPython,
                title: 'Model Layer (Python)',
                description: 'PyTorch-based Improved Character CRNN neural network for character recognition from Ranjana script images. Trained for 500 epochs on comprehensive dataset. Flask REST API serves the model for real-time inference with character segmentation and recognition.',
                bgColor: 'bg-blue-50',
                iconColor: 'text-blue-600',
                delay: 0.1
              },
              {
                icon: FaJava,
                title: 'Presenter Layer (Java)',
                description: 'Spring Boot microservices handle business logic, API orchestration, data validation, and admin dashboard. Manages OCR history, analytics, character statistics, and settings with in-memory storage for fast access.',
                bgColor: 'bg-orange-50',
                iconColor: 'text-orange-600',
                delay: 0.2
              },
              {
                icon: FaReact,
                title: 'View Layer (React)',
                description: 'Modern React 18 frontend with Vite, Tailwind CSS, and Framer Motion for intuitive user interface. Features responsive design, AR visualization, real-time OCR, and translation capabilities. Recharts for admin analytics dashboard.',
                bgColor: 'bg-cyan-50',
                iconColor: 'text-cyan-600',
                delay: 0.3
              }
            ].map((layer, index) => {
              const LayerIcon = layer.icon
              return (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  initial="rest"
                  whileHover="hover"
                  variants={cardHoverVariants}
                  className="flex items-start space-x-3 sm:space-x-4 p-3 sm:p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-300 group"
                >
                  <motion.div
                    animate={{ rotate: [0, 10, -10, 0] }}
                    transition={{ repeat: Infinity, duration: 3, delay: index * 0.5 }}
                    className={`p-2 sm:p-3 rounded-xl ${layer.bgColor} flex-shrink-0 group-hover:shadow-lg transition-shadow duration-300`}
                  >
                    <LayerIcon className={`text-2xl sm:text-3xl ${layer.iconColor}`} />
                  </motion.div>
                  <div className="flex-1">
                    <h3 className="font-bold text-base sm:text-lg text-gray-800 mb-1 sm:mb-2 group-hover:text-primary-600 transition-colors duration-300">
                      {layer.title}
                    </h3>
                    <p className="text-sm sm:text-base text-gray-700 leading-relaxed">
                      {layer.description}
                    </p>
                  </div>
                </motion.div>
              )
            })}
          </motion.div>
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
              { value: '80%', label: 'Training Split', color: 'text-green-600', bgColor: 'bg-green-50' },
              { value: '90%+', label: 'Accuracy', color: 'text-pink-600', bgColor: 'bg-pink-50' }
            ].map((metric, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ scale: 1.05, y: -5 }}
                className={`${metric.bgColor} p-3 sm:p-4 rounded-lg text-center group cursor-pointer`}
              >
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
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

        {/* Key Features - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="card mb-6 sm:mb-8 md:mb-12"
        >
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6"
          >
            <div className="p-2 sm:p-3 rounded-xl bg-green-100">
              <FaCheckCircle className="text-2xl sm:text-3xl text-green-600" />
            </div>
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">Key Features</h2>
          </motion.div>
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4"
          >
            {[
              'Character-level recognition with individual bounding boxes',
              'Ranjana script input to Devanagari output conversion',
              'Confidence scoring for each recognized character',
              'Word grouping based on character spacing',
              'AR overlay visualization similar to Google Lens',
              'Optional English translation support',
              'Camera capture and image upload support',
              'Admin dashboard with analytics and OCR history'
            ].map((feature, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                whileHover={{ scale: 1.02, x: 5 }}
                className="flex items-start space-x-2 sm:space-x-3 p-2 sm:p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-300 group"
              >
                <motion.span
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ repeat: Infinity, duration: 2, delay: index * 0.2 }}
                  className="text-green-600 text-lg sm:text-xl font-bold flex-shrink-0"
                >
                  ✓
                </motion.span>
                <span className="text-sm sm:text-base text-gray-700 group-hover:text-gray-900 transition-colors duration-300">
                  {feature}
                </span>
              </motion.div>
            ))}
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
              'Expand translation support to more languages including Hindi, Sanskrit, and regional languages',
              'Add text-to-speech functionality for accessibility and language learning',
              'Develop native mobile applications for iOS and Android platforms',
              'Support batch processing for large document archives and historical texts',
              'Integrate database storage for persistent OCR history and user management'
            ].map((goal, index) => (
              <motion.li
                key={index}
                variants={itemVariants}
                whileHover={{ scale: 1.02, x: 5 }}
                className="flex items-start space-x-2 sm:space-x-3 p-2 sm:p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-300 group"
              >
                <motion.span
                  animate={{ x: [0, 5, 0] }}
                  transition={{ repeat: Infinity, duration: 2, delay: index * 0.3 }}
                  className="text-primary-600 text-lg sm:text-xl font-bold flex-shrink-0 group-hover:scale-110 transition-transform duration-300"
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

        {/* Contact - Responsive with Animation */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-50px" }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="text-center card"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="mb-4 sm:mb-6"
          >
            <FaShieldAlt className="text-4xl sm:text-5xl text-primary-600 mx-auto mb-3 sm:mb-4" />
            <h2 className="text-xl sm:text-2xl font-bold mb-3 sm:mb-4 text-gray-800">Get Involved</h2>
            <p className="text-sm sm:text-base text-gray-700 mb-4 sm:mb-6 px-4">
              Lipika is an open-source project focused on preserving and digitizing Ranjana script. 
              Contributions, feedback, and suggestions are always welcome!
            </p>
          </motion.div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.7 }}
            className="flex flex-col sm:flex-row justify-center gap-3 sm:gap-4"
          >
            <motion.a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              className="btn-primary text-sm sm:text-base px-4 sm:px-6 py-2.5 sm:py-3 flex items-center justify-center space-x-2"
            >
              <FaCode />
              <span>View on GitHub</span>
            </motion.a>
            <motion.a 
              href="mailto:contact@lipika.com" 
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              className="btn-secondary text-sm sm:text-base px-4 sm:px-6 py-2.5 sm:py-3 flex items-center justify-center space-x-2"
            >
              <FaMobileAlt />
              <span>Contact Us</span>
            </motion.a>
          </motion.div>
        </motion.div>
      </main>
    </div>
  )
}

export default About
