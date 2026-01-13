import { useState, useEffect } from 'react'
import { Link, useLocation, Outlet, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { FaTachometerAlt, FaHistory, FaCog, FaBars, FaChartBar, FaFont, FaSignOutAlt, FaUsers, FaExclamationTriangle, FaEnvelope } from 'react-icons/fa'
import { ROUTES } from '../config/constants'
import { useAuth } from '../context/AuthContext'
import logoImage from '../images/Logo.png'
import ConfirmationModal from './ConfirmationModal'

const AdminLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768)
  const [showLogoutModal, setShowLogoutModal] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { logout } = useAuth()

  // Handle responsive sidebar on window resize
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setSidebarOpen(true)
      } else {
        setSidebarOpen(false)
      }
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const handleLogout = () => {
    setShowLogoutModal(true)
  }

  const confirmLogout = () => {
    logout()
    navigate('/login')
  }

  const cancelLogout = () => {
    setShowLogoutModal(false)
  }

  const menuItems = [
    { path: ROUTES.ADMIN_DASHBOARD, label: 'Dashboard', icon: FaTachometerAlt },
    { path: ROUTES.ADMIN_OCR_HISTORY, label: 'OCR History', icon: FaHistory },
    { path: ROUTES.ADMIN_ANALYTICS, label: 'Analytics', icon: FaChartBar },
    { path: ROUTES.ADMIN_CHARACTERS, label: 'Character Stats', icon: FaFont },
    { path: '/admin/users', label: 'User Management', icon: FaUsers },
    { path: '/admin/contacts', label: 'Contact Messages', icon: FaEnvelope },
    { path: ROUTES.ADMIN_SETTINGS, label: 'Settings', icon: FaCog },
  ]

  const isActive = (path) => {
    // Special case for dashboard: both /admin and /admin/dashboard should be active
    if (path === ROUTES.ADMIN_DASHBOARD) {
      return location.pathname === ROUTES.ADMIN || location.pathname === ROUTES.ADMIN_DASHBOARD
    }
    return location.pathname === path
  }

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  return (
    <div className="h-screen bg-gray-50 flex flex-col overflow-hidden">
      {/* Top Bar - Sticky */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50 flex-shrink-0 shadow-sm">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center space-x-3">
            {/* Hamburger Toggle Button - Always visible */}
            <button
              onClick={toggleSidebar}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
              aria-label="Toggle sidebar"
              title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            >
              <FaBars className="text-xl text-gray-700" />
            </button>
            
            {/* Logo and Text */}
            <Link to={ROUTES.HOME} className="flex items-center space-x-3 group">
              <img 
                src={logoImage} 
                alt="Lipika Logo" 
                className="h-8 w-auto"
              />
              <div>
                <h1 className="text-xl font-black text-primary-600">
                  Lipika
                </h1>
                <p className="text-xs text-gray-500 group-hover:text-gray-700 transition-colors">
                  Ranjana OCR System
                </p>
              </div>
            </Link>
          </div>
          <div className="flex items-center space-x-4">
            <button
              onClick={handleLogout}
              className="px-4 py-2 text-sm font-semibold text-gray-700 hover:text-white hover:bg-red-600 rounded-lg transition-all duration-200 flex items-center space-x-2 border border-gray-300 hover:border-red-600"
            >
              <FaSignOutAlt />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container - Flex with fixed height */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Sidebar Overlay for mobile - Outside sidebar */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Sidebar - Sticky, 100vh height, scrollable navigation */}
        <aside
          className={`bg-white border-r border-gray-200 transition-all duration-300 flex-shrink-0 flex flex-col shadow-lg
            ${sidebarOpen ? 'w-64' : 'w-0 md:w-0'}
            md:relative md:h-[calc(100vh-64px)]
            fixed top-0 left-0 h-screen z-40
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
          `}
        >
          {/* Sidebar Content - Fixed container, scrollable navigation */}
          <div className={`flex flex-col h-full w-64 transition-opacity duration-300 ${sidebarOpen ? 'opacity-100' : 'opacity-0 md:opacity-0'}`}>
            {/* Navigation - Scrollable if content is long */}
            <nav className="flex-1 p-4 space-y-2 overflow-y-auto custom-scrollbar mt-16 md:mt-0">
              {menuItems.map((item) => {
                const IconComponent = item.icon
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}
                    className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 whitespace-nowrap ${
                      isActive(item.path)
                        ? 'bg-primary-600 text-white shadow-md'
                        : 'text-gray-700 hover:bg-primary-100 hover:text-primary-700'
                    }`}
                    title={item.label}
                  >
                    <IconComponent className="text-xl flex-shrink-0" />
                    <span className="font-semibold">
                      {item.label}
                    </span>
                  </Link>
                )
              })}
            </nav>
            
            {/* Admin Panel Text at Bottom */}
            <div className="p-4 border-t border-gray-200 bg-white/50">
              <div className="text-center">
                <p className="text-xs font-semibold text-primary-600 uppercase tracking-wider">
                  Admin Panel
                </p>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content Area - Scrollable */}
        <main className="flex-1 overflow-y-auto transition-all duration-300 custom-scrollbar bg-gray-50 w-full">
          <div className="p-4 sm:p-6 max-w-full">
            <Outlet />
          </div>
        </main>
      </div>

      {/* Logout Confirmation Modal */}
      <ConfirmationModal
        isOpen={showLogoutModal}
        onClose={cancelLogout}
        title="Confirm Logout"
        message="Are you sure you want to logout? You will need to login again to access the admin panel."
        icon={<FaExclamationTriangle className="text-3xl text-red-600" />}
        iconBgColor="bg-red-100"
        confirmText="Logout"
        confirmIcon={FaSignOutAlt}
        confirmClassName="bg-red-600 hover:bg-red-700"
        onConfirm={confirmLogout}
      />
    </div>
  )
}

export default AdminLayout
