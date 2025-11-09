import { useState } from 'react'
import { Link, useLocation, Outlet } from 'react-router-dom'
import { ROUTES } from '../config/constants'

const AdminLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const location = useLocation()

  const menuItems = [
    { path: '/admin/dashboard', label: 'Dashboard', icon: '📊' },
    { path: '/admin/ocr-history', label: 'OCR History', icon: '📜' },
    { path: '/admin/settings', label: 'Settings', icon: '⚙️' },
  ]

  const isActive = (path) => location.pathname === path

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Bar */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <span className="text-2xl">☰</span>
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
      </div>

      <div className="flex">
        {/* Sidebar */}
        <aside
          className={`bg-white border-r border-gray-200 transition-all duration-300 ${
            sidebarOpen ? 'w-64' : 'w-0'
          } overflow-hidden`}
        >
          <nav className="p-4 space-y-2">
            {menuItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive(item.path)
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <span className="text-xl">{item.icon}</span>
                <span className="font-semibold">{item.label}</span>
              </Link>
            ))}
          </nav>
        </aside>

        {/* Main Content */}
        <main className={`flex-1 transition-all duration-300 ${sidebarOpen ? 'ml-0' : 'ml-0'}`}>
          <div className="p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

export default AdminLayout

