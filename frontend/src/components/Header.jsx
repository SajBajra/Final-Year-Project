import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { FaScroll, FaUser, FaSignOutAlt, FaCog, FaBars, FaTimes } from 'react-icons/fa'
import { useAuth } from '../context/AuthContext'

const Header = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { isAuthenticated, user, logout, isAdmin } = useAuth()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const isActive = (path) => location.pathname === path

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/features', label: 'Features' },
    { path: '/about', label: 'About' }
  ]

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
          
          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item, index) => (
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
            
            {isAuthenticated() ? (
              <div className="ml-4 flex items-center gap-2">
                {isAdmin() && (
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => navigate('/admin')}
                    className="p-2 bg-purple-100 hover:bg-purple-200 rounded-lg text-purple-600 transition-colors"
                    title="Admin Dashboard"
                  >
                    <FaCog />
                  </motion.button>
                )}
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  className="flex items-center gap-2 px-3 py-2 bg-primary-100 rounded-lg"
                >
                  <FaUser className="text-primary-600" />
                  <span className="text-sm font-medium text-primary-700">{user?.username}</span>
                </motion.div>
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    logout();
                    navigate('/');
                  }}
                  className="p-2 bg-red-100 hover:bg-red-200 rounded-lg text-red-600 transition-colors"
                  title="Logout"
                >
                  <FaSignOutAlt />
                </motion.button>
              </div>
            ) : (
              <div className="ml-4 flex items-center gap-2">
                <Link to="/login" className="btn-outline text-sm px-4 py-2">
                  Login
                </Link>
                <Link to="/register" className="btn-primary text-sm px-4 py-2">
                  Register
                </Link>
              </div>
            )}
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg text-gray-600 hover:bg-gray-100 transition-colors"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <FaTimes className="text-xl" /> : <FaBars className="text-xl" />}
          </button>
        </div>

        {/* Mobile Navigation Menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <>
              {/* Overlay with backdrop blur */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                onClick={() => setMobileMenuOpen(false)}
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden"
              />
              
              {/* Menu - Slide from right with better styling */}
              <motion.nav
                initial={{ x: '100%', opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                exit={{ x: '100%', opacity: 0 }}
                transition={{ 
                  type: 'spring', 
                  damping: 30, 
                  stiffness: 300,
                  mass: 0.8
                }}
                className="fixed top-0 right-0 h-full w-80 max-w-[85vw] bg-white shadow-2xl z-50 md:hidden overflow-y-auto"
              >
                {/* Menu Header */}
                <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between z-10">
                  <h2 className="text-lg font-bold text-gray-900">Menu</h2>
                  <button
                    onClick={() => setMobileMenuOpen(false)}
                    className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
                    aria-label="Close menu"
                  >
                    <FaTimes className="text-xl" />
                  </button>
                </div>

                {/* Menu Content */}
                <div className="p-6 space-y-1">
                  {/* Navigation Links */}
                  {navItems.map((item, index) => (
                    <motion.div
                      key={item.path}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05, duration: 0.3 }}
                    >
                      <Link
                        to={item.path}
                        onClick={() => setMobileMenuOpen(false)}
                        className={`flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all ${
                          isActive(item.path)
                            ? 'bg-gradient-to-r from-blue-50 to-purple-50 text-blue-600 font-semibold shadow-sm'
                            : 'text-gray-700 hover:bg-gray-50 active:bg-gray-100'
                        }`}
                      >
                        <span className="text-lg">{item.label}</span>
                      </Link>
                    </motion.div>
                  ))}
                  
                  {/* Divider */}
                  <div className="border-t border-gray-200 my-6"></div>
                  
                  {/* Auth Section */}
                  {isAuthenticated() ? (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="space-y-2"
                    >
                      {isAdmin() && (
                        <button
                          onClick={() => {
                            navigate('/admin');
                            setMobileMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-gray-700 hover:bg-purple-50 hover:text-purple-600 transition-all active:scale-95"
                        >
                          <FaCog className="text-lg" />
                          <span className="font-medium">Admin Dashboard</span>
                        </button>
                      )}
                      <div className="flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-primary-50 to-blue-50 rounded-xl border border-primary-100">
                        <div className="p-2 bg-primary-100 rounded-lg">
                          <FaUser className="text-primary-600" />
                        </div>
                        <div className="flex-1">
                          <p className="text-xs text-gray-500">Logged in as</p>
                          <p className="text-sm font-semibold text-primary-700">{user?.username}</p>
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          logout();
                          navigate('/');
                          setMobileMenuOpen(false);
                        }}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-red-600 hover:bg-red-50 transition-all active:scale-95 font-medium"
                      >
                        <FaSignOutAlt className="text-lg" />
                        <span>Logout</span>
                      </button>
                    </motion.div>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.2 }}
                      className="space-y-3"
                    >
                      <Link
                        to="/login"
                        onClick={() => setMobileMenuOpen(false)}
                        className="block w-full px-4 py-3 rounded-xl border-2 border-gray-300 text-center font-medium text-gray-700 hover:bg-gray-50 hover:border-gray-400 transition-all active:scale-95"
                      >
                        Login
                      </Link>
                      <Link
                        to="/register"
                        onClick={() => setMobileMenuOpen(false)}
                        className="block w-full px-4 py-3 rounded-xl bg-gradient-to-r from-primary-600 to-purple-600 text-white text-center font-semibold hover:from-primary-700 hover:to-purple-700 transition-all active:scale-95 shadow-lg"
                      >
                        Register
                      </Link>
                    </motion.div>
                  )}
                </div>
              </motion.nav>
            </>
          )}
        </AnimatePresence>
      </div>
    </header>
  )
}

export default Header

