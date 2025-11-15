import { useState } from 'react'
import { Link, useLocation, Outlet } from 'react-router-dom'
import { FaChartLine, FaHistory, FaCog, FaBars, FaTimes, FaChartBar, FaFont, FaChevronLeft, FaChevronRight } from 'react-icons/fa'
import { ROUTES } from '../config/constants'

const AdminLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const location = useLocation()

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
    <div className="h-screen bg-gray-50 flex flex-col overflow-hidden">
      {/* Top Bar - Sticky */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50 flex-shrink-0">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center space-x-4">
            {/* Mobile Toggle Button */}
            <button
              onClick={toggleSidebar}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors md:hidden"
              aria-label="Toggle sidebar"
            >
              {sidebarOpen ? <FaTimes className="text-2xl" /> : <FaBars className="text-2xl" />}
            </button>
            
            <h1 className="text-2xl font-bold text-primary-600">Admin Panel</h1>
          </div>
          <div className="flex items-center space-x-4">
            <Link
              to={ROUTES.HOME}
              className="px-4 py-2 text-sm font-semibold text-secondary-500 hover:text-primary-600 transition-colors"
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </header>

      {/* Main Container - Flex with fixed height */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Sidebar - Sticky, 100vh height, scrollable navigation */}
        <aside
          className={`bg-white border-r border-gray-200 transition-all duration-300 flex-shrink-0 flex flex-col ${
            sidebarOpen ? 'w-64' : 'w-0'
          } ${
            // Desktop: sticky sidebar, Mobile: fixed overlay
            'md:sticky md:top-16 md:h-[calc(100vh-64px)] fixed top-16 h-[calc(100vh-64px)] z-40'
          } overflow-hidden`}
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
            {/* Desktop Toggle Button - Inside Sidebar */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
              <div className={`overflow-hidden transition-all duration-300 ${sidebarOpen ? 'w-full' : 'w-0'}`}>
                <h2 className="text-lg font-bold text-gray-800 whitespace-nowrap">Navigation</h2>
              </div>
              <button
                onClick={toggleSidebar}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors flex-shrink-0 hidden md:flex items-center justify-center"
                aria-label="Toggle sidebar"
                title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
              >
                {sidebarOpen ? (
                  <FaChevronLeft className="text-xl text-gray-600" />
                ) : (
                  <FaChevronRight className="text-xl text-gray-600" />
                )}
              </button>
            </div>

            {/* Navigation - Scrollable if content is long */}
            <nav className={`flex-1 p-4 space-y-2 overflow-y-auto custom-scrollbar ${sidebarOpen ? 'w-full' : 'w-0'} overflow-hidden`}>
              {menuItems.map((item) => {
                const IconComponent = item.icon
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}
                    className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors whitespace-nowrap ${
                      isActive(item.path)
                        ? 'bg-primary-600 text-white'
                        : 'text-gray-700 hover:bg-gray-100'
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
          </div>
        </aside>

        {/* Desktop Toggle Button - Outside Sidebar (when collapsed) */}
        {!sidebarOpen && (
          <button
            onClick={toggleSidebar}
            className="absolute left-0 top-1/2 transform -translate-y-1/2 z-50 p-2 bg-white border border-gray-200 rounded-r-lg shadow-md hover:bg-gray-50 transition-colors hidden md:flex items-center justify-center"
            aria-label="Expand sidebar"
            title="Expand sidebar"
          >
            <FaChevronRight className="text-xl text-gray-600" />
          </button>
        )}

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
