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
              {/* Overlay */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setMobileMenuOpen(false)}
                className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
              />
              
              {/* Menu */}
              <motion.nav
                initial={{ x: '100%' }}
                animate={{ x: 0 }}
                exit={{ x: '100%' }}
                transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                className="fixed top-16 right-0 h-[calc(100vh-4rem)] w-64 bg-white shadow-xl z-50 md:hidden overflow-y-auto"
              >
                <div className="p-4 space-y-2">
                  {navItems.map((item) => (
                    <Link
                      key={item.path}
                      to={item.path}
                      onClick={() => setMobileMenuOpen(false)}
                      className={`block px-4 py-3 rounded-lg font-medium transition-colors ${
                        isActive(item.path)
                          ? 'bg-blue-100 text-blue-600 font-semibold'
                          : 'text-gray-600 hover:bg-gray-100'
                      }`}
                    >
                      {item.label}
                    </Link>
                  ))}
                  
                  <div className="border-t border-gray-200 my-4"></div>
                  
                  {isAuthenticated() ? (
                    <div className="space-y-2">
                      {isAdmin() && (
                        <button
                          onClick={() => {
                            navigate('/admin');
                            setMobileMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-gray-600 hover:bg-gray-100 transition-colors"
                        >
                          <FaCog />
                          <span>Admin Dashboard</span>
                        </button>
                      )}
                      <div className="flex items-center gap-3 px-4 py-3 bg-primary-50 rounded-lg">
                        <FaUser className="text-primary-600" />
                        <span className="text-sm font-medium text-primary-700">{user?.username}</span>
                      </div>
                      <button
                        onClick={() => {
                          logout();
                          navigate('/');
                          setMobileMenuOpen(false);
                        }}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-red-600 hover:bg-red-50 transition-colors"
                      >
                        <FaSignOutAlt />
                        <span>Logout</span>
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Link
                        to="/login"
                        onClick={() => setMobileMenuOpen(false)}
                        className="block px-4 py-3 rounded-lg border border-gray-300 text-center font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        Login
                      </Link>
                      <Link
                        to="/register"
                        onClick={() => setMobileMenuOpen(false)}
                        className="block px-4 py-3 rounded-lg bg-primary-600 text-white text-center font-medium hover:bg-primary-700 transition-colors"
                      >
                        Register
                      </Link>
                    </div>
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

