import { useState, useEffect } from 'react'
import { FaChartBar, FaBullseye, FaFont, FaChartLine } from 'react-icons/fa'
import { getDashboardStats } from '../../services/adminService'

const AdminDashboard = () => {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    try {
      setLoading(true)
      const data = await getDashboardStats()
      setStats(data)
    } catch (error) {
      console.error('Error loading stats:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-600 border-t-transparent"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800">Dashboard</h2>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Total Records</p>
              <p className="text-3xl font-bold text-gray-800 mt-2">
                {stats?.totalRecords || 0}
              </p>
            </div>
            <FaChartBar className="text-4xl text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Avg Confidence</p>
              <p className="text-3xl font-bold text-gray-800 mt-2">
                {stats?.avgConfidence ? `${stats.avgConfidence.toFixed(1)}%` : '0%'}
              </p>
            </div>
            <FaBullseye className="text-4xl text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Total Characters</p>
              <p className="text-3xl font-bold text-gray-800 mt-2">
                {stats?.totalCharacters || 0}
              </p>
            </div>
            <FaFont className="text-4xl text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Recent Activity (24h)</p>
              <p className="text-3xl font-bold text-gray-800 mt-2">
                {stats?.recentActivity || 0}
              </p>
            </div>
            <FaChartLine className="text-4xl text-primary-600" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default AdminDashboard

