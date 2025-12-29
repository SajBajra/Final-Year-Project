import { useState } from 'react'
import { Link, useLocation, Outlet, useNavigate } from 'react-router-dom'
import { FaChartLine, FaHistory, FaCog, FaBars, FaChartBar, FaFont, FaSignOutAlt } from 'react-icons/fa'
import { ROUTES } from '../config/constants'
import { useAuth } from '../context/AuthContext'
import logoImage from '../images/Logo.png'

const AdminLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const location = useLocation()
  const navigate = useNavigate()
  const { logout } = useAuth()

  const handleLogout = () => {
    logout()
    navigate(ROUTES.ADMIN_LOGIN)
  }

  const menuItems = [
    { path: ROUTES.ADMIN_DASHBOARD, label: 'Dashboard', icon: FaChartLine },
    { path: ROUTES.ADMIN_OCR_HISTORY, label: 'OCR History', icon: FaHistory },
    { path: ROUTES.ADMIN_ANALYTICS, label: 'Analytics', icon: FaChartBar },
    { path: ROUTES.ADMIN_CHARACTERS, label: 'Character Stats', icon: FaFont },
    { path: ROUTES.ADMIN_SETTINGS, label: 'Settings', icon: FaCog },
  ]

  const isActive = (path) => location.pathname === path

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  return (
    <div className="h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 flex flex-col overflow-hidden">
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
              className="px-4 py-2 text-sm font-semibold text-error-600 hover:text-error-700 hover:bg-error-50 rounded-lg transition-colors flex items-center space-x-2"
            >
              <FaSignOutAlt />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Container - Flex with fixed height */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Sidebar - Sticky, 100vh height, scrollable navigation */}
        <aside
          className={`bg-gradient-to-b from-white via-primary-50/30 to-secondary-50/30 border-r border-gray-200 transition-all duration-300 flex-shrink-0 flex flex-col ${
            sidebarOpen ? 'w-64' : 'w-0'
          } ${
            // Desktop: sticky sidebar, Mobile: fixed overlay
            'md:sticky md:top-16 md:h-[calc(100vh-64px)] fixed top-16 h-[calc(100vh-64px)] z-40'
          } overflow-hidden shadow-lg`}
        >
          {/* Sidebar Overlay for mobile */}
          {sidebarOpen && (
            <div
              className="fixed inset-0 bg-black bg-opacity-50 z-30 md:hidden"
              onClick={() => setSidebarOpen(false)}
            />
          )}
          
          {/* Sidebar Content - Fixed container, scrollable navigation */}
          <div className={`flex flex-col h-full ${sidebarOpen ? 'w-64' : 'w-0'} overflow-hidden transition-all duration-300`}>
            {/* Navigation - Scrollable if content is long */}
            <nav className={`flex-1 p-4 space-y-2 overflow-y-auto custom-scrollbar ${sidebarOpen ? 'w-full' : 'w-0'} overflow-hidden`}>
              {menuItems.map((item, index) => {
                const IconComponent = item.icon
                // Alternate colors for better visualization
                const colors = [
                  { active: 'bg-primary-600 text-white', hover: 'hover:bg-primary-100 hover:text-primary-700' },
                  { active: 'bg-secondary-500 text-white', hover: 'hover:bg-secondary-100 hover:text-secondary-700' },
                  { active: 'bg-primary-600 text-white', hover: 'hover:bg-primary-100 hover:text-primary-700' },
                  { active: 'bg-secondary-500 text-white', hover: 'hover:bg-secondary-100 hover:text-secondary-700' },
                  { active: 'bg-primary-600 text-white', hover: 'hover:bg-primary-100 hover:text-primary-700' },
                ]
                const colorScheme = colors[index % colors.length]
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}
                    className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 whitespace-nowrap ${
                      isActive(item.path)
                        ? `${colorScheme.active} shadow-md`
                        : `text-gray-700 ${colorScheme.hover}`
                    }`}
                    title={item.label}
                  >
                    <IconComponent className="text-xl flex-shrink-0" />
                    <span className={`font-semibold transition-all duration-300 overflow-hidden ${
                      sidebarOpen ? 'opacity-100 w-auto' : 'opacity-0 w-0'
                    }`}>
                      {item.label}
                    </span>
                  </Link>
                )
              })}
            </nav>
            
            {/* Admin Panel Text at Bottom */}
            <div className={`p-4 border-t border-gray-200 bg-white/50 ${sidebarOpen ? 'w-full' : 'w-0'} overflow-hidden`}>
              <div className={`text-center transition-all duration-300 ${
                sidebarOpen ? 'opacity-100' : 'opacity-0'
              }`}>
                <p className="text-xs font-semibold text-primary-600 uppercase tracking-wider">
                  Admin Panel
                </p>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content Area - Scrollable */}
        <main 
          className={`flex-1 overflow-y-auto transition-all duration-300 custom-scrollbar ${
            // Add margin on desktop when sidebar is open
            sidebarOpen ? 'md:ml-0' : 'md:ml-0'
          }`}
        >
          <div className="p-4 sm:p-6 max-w-full">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

export default AdminLayout
