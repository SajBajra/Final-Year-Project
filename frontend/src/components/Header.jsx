import { useState, useRef, useEffect } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { FaScroll, FaUser, FaSignOutAlt, FaCog, FaBars, FaTimes, FaChevronDown, FaTachometerAlt } from 'react-icons/fa'
import { useAuth } from '../context/AuthContext'
import logoImage from '../images/Logo.png'

const Header = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { isAuthenticated, user, logout, isAdmin } = useAuth()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false)
  const profileDropdownRef = useRef(null)

  const isActive = (path) => location.pathname === path

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/features', label: 'Features' },
    { path: '/about', label: 'About' }
  ]

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (profileDropdownRef.current && !profileDropdownRef.current.contains(event.target)) {
        setProfileDropdownOpen(false)
      }
    }

    if (profileDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [profileDropdownOpen])

  return (
    <>
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 border-b border-white/20 shadow-sm">
        <div className="container mx-auto px-4 py-4 max-w-7xl">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center space-x-3 group">
              <img 
                src={logoImage} 
                alt="Lipika Logo" 
                className="h-10 w-auto"
              />
              <div>
                <h1 className="text-2xl font-black text-primary-600">
                  Lipika
                </h1>
                <p className="text-xs text-gray-500 group-hover:text-gray-700 transition-colors">
                  Ranjana OCR System
                </p>
              </div>
            </Link>
            
            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center flex-1 justify-center">
              <nav className="flex items-center space-x-8">
                {navItems.map((item) => (
                  <Link 
                    key={item.path}
                    to={item.path}
                    className="relative group"
                  >
                    <span className={`font-semibold text-base transition-colors duration-300 ${
                      isActive(item.path)
                        ? 'text-primary-600'
                        : 'text-gray-700 group-hover:text-primary-600'
                    }`}>
                      {item.label}
                    </span>
                    
                    {/* Active Underline */}
                    {isActive(item.path) && (
                      <motion.div
                        layoutId="activeNav"
                        className="absolute -bottom-1 left-0 right-0 h-0.5 bg-primary-600 rounded-full"
                        transition={{ type: "spring", stiffness: 380, damping: 30 }}
                      />
                    )}
                    
                    {/* Hover Underline */}
                    {!isActive(item.path) && (
                      <span className="absolute -bottom-1 left-0 right-0 h-0.5 bg-primary-600 rounded-full scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left" />
                    )}
                  </Link>
                ))}
              </nav>
            </div>
            
            <div className="hidden md:block">
              {isAuthenticated() ? (
                <div className="relative" ref={profileDropdownRef}>
                <button
                  onClick={() => setProfileDropdownOpen(!profileDropdownOpen)}
                  className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
                >
                  <FaUser className="text-gray-600 text-sm" />
                  <span className="text-sm font-medium text-gray-700">{user?.username}</span>
                  <FaChevronDown 
                    className={`text-gray-500 text-xs transition-transform duration-200 ${
                      profileDropdownOpen ? 'rotate-180' : ''
                    }`} 
                  />
                </button>

                {/* Profile Dropdown Menu */}
                <AnimatePresence>
                  {profileDropdownOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                      className="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50"
                    >
                      {/* User Info */}
                      <div className="px-4 py-3 border-b border-gray-200">
                        <p className="text-sm font-semibold text-gray-900">{user?.username}</p>
                        <p className="text-xs text-black mt-0.5">{user?.email}</p>
                        <p className="text-xs font-medium mt-1">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${
                            user?.role === 'PREMIUM' || isAdmin() 
                              ? 'bg-gradient-to-r from-yellow-400 to-yellow-500 text-gray-900' 
                              : 'bg-gray-200 text-gray-700'
                          }`}>
                            {user?.role === 'PREMIUM' || isAdmin() ? '✨ Paid Account' : 'Free Account'}
                          </span>
                        </p>
                      </div>

                      {/* Profile (Regular Users) */}
                      {!isAdmin() && (
                        <button
                          onClick={() => {
                            navigate('/profile');
                            setProfileDropdownOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-black hover:bg-gray-50 transition-colors"
                        >
                          <FaUser className="text-primary-600" />
                          <span>My Profile</span>
                        </button>
                      )}

                      {/* Admin Dashboard (Admin only) */}
                      {isAdmin() && (
                        <button
                          onClick={() => {
                            navigate('/admin');
                            setProfileDropdownOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-black hover:bg-gray-50 transition-colors"
                        >
                          <FaTachometerAlt className="text-primary-600" />
                          <span>Admin Dashboard</span>
                        </button>
                      )}

                      {/* Settings (Admin only) */}
                      {isAdmin() && (
                        <button
                          onClick={() => {
                            navigate('/admin/settings');
                            setProfileDropdownOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-black hover:bg-gray-50 transition-colors"
                        >
                          <FaCog className="text-primary-600" />
                          <span>Settings</span>
                        </button>
                      )}

                      {/* Logout */}
                      <button
                        onClick={() => {
                          logout();
                          navigate('/');
                          setProfileDropdownOpen(false);
                        }}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                      >
                        <FaSignOutAlt />
                        <span>Logout</span>
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              ) : (
                <div className="flex items-center gap-3">
                  <Link
                    to="/login"
                    className="btn-secondary btn-md"
                  >
                    Login
                  </Link>
                  <Link
                    to="/register"
                    className="btn-primary btn-md"
                  >
                    Sign Up
                  </Link>
                </div>
              )}
            </div>

            {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2.5 rounded-lg text-gray-700 hover:bg-gray-100 active:bg-gray-200 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
            aria-label="Toggle menu"
            aria-expanded={mobileMenuOpen}
          >
            <motion.div
              animate={{ rotate: mobileMenuOpen ? 90 : 0 }}
              transition={{ duration: 0.2 }}
            >
              {mobileMenuOpen ? (
                <FaTimes className="text-xl text-gray-700" />
              ) : (
                <FaBars className="text-xl text-gray-700" />
              )}
            </motion.div>
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Navigation Menu - Offcanvas Style */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <>
            {/* Backdrop/Overlay */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={() => setMobileMenuOpen(false)}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[60] md:hidden"
              aria-hidden="true"
            />
            
            {/* Offcanvas Menu */}
            <motion.aside
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ 
                type: 'tween',
                duration: 0.3,
                ease: [0.4, 0, 0.2, 1]
              }}
              className="fixed top-0 right-0 h-full w-80 max-w-[85vw] bg-white shadow-2xl z-[70] md:hidden flex flex-col"
            >
              {/* Header with Close Button */}
              <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-gray-50">
                <h2 className="text-lg font-semibold text-gray-900">Menu</h2>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-200 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
                  aria-label="Close menu"
                >
                  <FaTimes className="text-xl" />
                </button>
              </div>

              {/* Scrollable Content */}
              <div className="flex-1 overflow-y-auto">
                <nav className="p-4">
                  {/* Navigation Links */}
                  <div className="space-y-2 mb-4">
                    {navItems.map((item, index) => (
                      <motion.div
                        key={item.path}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05, duration: 0.2 }}
                      >
                        <Link
                          to={item.path}
                          onClick={() => setMobileMenuOpen(false)}
                          className={`relative block px-4 py-3 rounded-lg font-semibold transition-all duration-300 ${
                            isActive(item.path)
                              ? 'bg-primary-600 text-white shadow-md'
                              : 'text-gray-700 hover:bg-gray-100 hover:translate-x-1'
                          }`}
                        >
                          {item.label}
                          {isActive(item.path) && (
                            <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-white rounded-r-full" />
                          )}
                        </Link>
                      </motion.div>
                    ))}
                  </div>

                  {/* Divider */}
                  <div className="border-t border-gray-200 my-4" />

                  {/* Auth Section */}
                  {isAuthenticated() ? (
                    <div className="space-y-3">
                      {/* User Info Card */}
                      <div className="bg-primary-50 rounded-lg p-4 border border-primary-100">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-full bg-primary-600 flex items-center justify-center">
                            <FaUser className="text-white text-sm" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-gray-900 truncate">
                              {user?.username}
                            </p>
                        <p className="text-xs text-black">
                          {isAdmin() ? 'Administrator' : 'User'}
                        </p>
                          </div>
                        </div>
                      </div>

                      {/* Profile Button (Regular Users) */}
                      {!isAdmin() && (
                        <button
                          onClick={() => {
                            navigate('/profile');
                            setMobileMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-black bg-gray-50 hover:bg-gray-100 border border-gray-200 transition-colors font-medium"
                        >
                          <FaUser className="text-primary-600" />
                          <span>My Profile</span>
                        </button>
                      )}
                      
                      {/* Admin Dashboard Button */}
                      {isAdmin() && (
                        <button
                          onClick={() => {
                            navigate('/admin');
                            setMobileMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-black bg-gray-50 hover:bg-gray-100 border border-gray-200 transition-colors font-medium"
                        >
                          <FaTachometerAlt className="text-primary-600" />
                          <span>Admin Dashboard</span>
                        </button>
                      )}

                      {/* Logout Button */}
                      <button
                        onClick={() => {
                          logout();
                          navigate('/');
                          setMobileMenuOpen(false);
                        }}
                        className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-white bg-red-600 hover:bg-red-700 transition-colors font-medium shadow-sm"
                      >
                        <FaSignOutAlt />
                        <span>Logout</span>
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {/* Login Button */}
                      <Link
                        to="/login"
                        onClick={() => setMobileMenuOpen(false)}
                        className="btn-secondary btn-lg w-full"
                      >
                        <FaUser />
                        <span>Login</span>
                      </Link>

                      {/* Sign Up Button */}
                      <Link
                        to="/register"
                        onClick={() => setMobileMenuOpen(false)}
                        className="btn-primary btn-lg w-full"
                      >
                        <FaUser />
                        <span>Sign Up</span>
                      </Link>
                    </div>
                  )}
                </nav>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  )
}

export default Header

