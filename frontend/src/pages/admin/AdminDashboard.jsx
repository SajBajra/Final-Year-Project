import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FaChartBar, FaBullseye, FaFont, FaChartLine, FaChartPie } from 'react-icons/fa'
import { 
  LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, BarChart, Bar 
} from 'recharts'
import { getDashboardStats, getAnalytics } from '../../services/adminService'

const AdminDashboard = () => {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analytics, setAnalytics] = useState(null)
  const [chartLoading, setChartLoading] = useState(false)

  const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1']

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      setLoading(true)
      setChartLoading(true)
      
      // Load stats and analytics in parallel
      const [statsData, analyticsResponse] = await Promise.all([
        getDashboardStats(),
        getAnalytics('daily', 7).catch(() => ({ success: false, data: null }))
      ])
      
      setStats(statsData)
      if (analyticsResponse.success) {
        setAnalytics(analyticsResponse.data)
      }
    } catch (error) {
      console.error('Error loading dashboard data:', error)
    } finally {
      setLoading(false)
      setChartLoading(false)
    }
  }

  // Format recent activity line chart data (last 7 days)
  const formatActivityChartData = () => {
    if (!analytics || !analytics.timeSeries) {
      // Generate empty data for last 7 days if no data
      const days = []
      const today = new Date()
      for (let i = 6; i >= 0; i--) {
        const date = new Date(today)
        date.setDate(date.getDate() - i)
        days.push({
          date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          requests: 0
        })
      }
      return days
    }
    
    const timeSeries = analytics.timeSeries || {}
    const sortedKeys = Object.keys(timeSeries).sort()
    const last7Days = sortedKeys.slice(-7)
    
    return last7Days.map(key => ({
      date: new Date(key).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      requests: timeSeries[key] || 0
    }))
  }

  // Format confidence distribution for pie chart
  const formatConfidenceDistribution = () => {
    if (!analytics || !analytics.confidenceDistribution) {
      return [
        { name: '90-100%', value: 0, color: '#10b981' },
        { name: '80-90%', value: 0, color: '#3b82f6' },
        { name: '70-80%', value: 0, color: '#f59e0b' },
        { name: '60-70%', value: 0, color: '#ef4444' },
        { name: 'Below 60%', value: 0, color: '#ef4444' }
      ]
    }
    
    const dist = analytics.confidenceDistribution || {}
    return [
      { name: '90-100%', value: dist['90-100%'] || 0, color: '#10b981' },
      { name: '80-90%', value: dist['80-90%'] || 0, color: '#3b82f6' },
      { name: '70-80%', value: dist['70-80%'] || 0, color: '#f59e0b' },
      { name: '60-70%', value: dist['60-70%'] || 0, color: '#ef4444' },
      { name: 'Below 60%', value: dist['Below 60%'] || 0, color: '#ef4444' }
    ].filter(item => item.value > 0)
  }

  // Format character count distribution (if available)
  const formatCharacterCountData = () => {
    if (!analytics || !analytics.characterSeries) {
      return []
    }
    
    const charSeries = analytics.characterSeries || {}
    const sortedKeys = Object.keys(charSeries).sort()
    const last7Days = sortedKeys.slice(-7)
    
    return last7Days.map(key => ({
      date: new Date(key).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      characters: charSeries[key] || 0
    }))
  }

  const activityChartData = formatActivityChartData()
  const confidencePieData = formatConfidenceDistribution()
  const characterChartData = formatCharacterCountData()
  const totalConfidenceRecords = confidencePieData.reduce((sum, item) => sum + item.value, 0)

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

  return (
    <div className="space-y-4 sm:space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-800">Dashboard</h2>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={loadDashboardData}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors duration-200 text-sm sm:text-base"
        >
          Refresh
        </motion.button>
      </motion.div>

      {/* Stats Grid - Responsive */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6"
      >
        <motion.div variants={itemVariants} className="bg-white rounded-lg p-4 sm:p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-xs sm:text-sm text-gray-500 font-semibold">Total Records</p>
              {loading ? (
                <div className="h-8 sm:h-10 w-16 sm:w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <p className="text-2xl sm:text-3xl font-bold text-gray-800 mt-2">
                  {stats?.totalRecords ?? 0}
                </p>
              )}
            </div>
            <div className="p-3 rounded-xl bg-blue-100">
              <FaChartBar className="text-2xl sm:text-3xl md:text-4xl text-primary-600" />
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="bg-white rounded-lg p-4 sm:p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-xs sm:text-sm text-gray-500 font-semibold">Avg Confidence</p>
              {loading ? (
                <div className="h-8 sm:h-10 w-16 sm:w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <p className="text-2xl sm:text-3xl font-bold text-gray-800 mt-2">
                  {stats?.avgConfidence ? `${(stats.avgConfidence * 100).toFixed(1)}%` : '0%'}
                </p>
              )}
            </div>
            <div className="p-3 rounded-xl bg-green-100">
              <FaBullseye className="text-2xl sm:text-3xl md:text-4xl text-green-600" />
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="bg-white rounded-lg p-4 sm:p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-xs sm:text-sm text-gray-500 font-semibold">Total Characters</p>
              {loading ? (
                <div className="h-8 sm:h-10 w-16 sm:w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <p className="text-2xl sm:text-3xl font-bold text-gray-800 mt-2">
                  {stats?.totalCharacters ?? 0}
                </p>
              )}
            </div>
            <div className="p-3 rounded-xl bg-purple-100">
              <FaFont className="text-2xl sm:text-3xl md:text-4xl text-purple-600" />
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="bg-white rounded-lg p-4 sm:p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow duration-200">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-xs sm:text-sm text-gray-500 font-semibold">Recent Activity (24h)</p>
              {loading ? (
                <div className="h-8 sm:h-10 w-16 sm:w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <p className="text-2xl sm:text-3xl font-bold text-gray-800 mt-2">
                  {stats?.recentActivity ?? 0}
                </p>
              )}
            </div>
            <div className="p-3 rounded-xl bg-orange-100">
              <FaChartLine className="text-2xl sm:text-3xl md:text-4xl text-orange-600" />
            </div>
          </div>
        </motion.div>
      </motion.div>

      {/* Charts Grid - Responsive */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6"
      >
        {/* Recent Activity Line Chart */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="bg-white rounded-lg p-4 sm:p-6 shadow-sm border border-gray-200"
        >
          <div className="flex items-center space-x-2 sm:space-x-3 mb-4 sm:mb-6">
            <div className="p-2 sm:p-3 rounded-xl bg-primary-100">
              <FaChartLine className="text-xl sm:text-2xl text-primary-600" />
            </div>
            <div>
              <h3 className="text-lg sm:text-xl font-bold text-gray-800">Recent Activity (7 Days)</h3>
              <p className="text-xs sm:text-sm text-gray-500">OCR requests over time</p>
            </div>
          </div>
          {chartLoading ? (
            <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={activityChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  dataKey="date" 
                  stroke="#6b7280"
                  style={{ fontSize: '12px' }}
                />
                <YAxis 
                  stroke="#6b7280"
                  style={{ fontSize: '12px' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Line 
                  type="monotone" 
                  dataKey="requests" 
                  stroke="#2563eb" 
                  strokeWidth={3}
                  dot={{ fill: '#2563eb', r: 5 }}
                  activeDot={{ r: 7 }}
                  name="OCR Requests"
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </motion.div>

        {/* Confidence Distribution Pie Chart */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-lg p-4 sm:p-6 shadow-sm border border-gray-200"
        >
          <div className="flex items-center space-x-2 sm:space-x-3 mb-4 sm:mb-6">
            <div className="p-2 sm:p-3 rounded-xl bg-green-100">
              <FaChartPie className="text-xl sm:text-2xl text-green-600" />
            </div>
            <div>
              <h3 className="text-lg sm:text-xl font-bold text-gray-800">Confidence Distribution</h3>
              <p className="text-xs sm:text-sm text-gray-500">Accuracy levels of OCR results</p>
            </div>
          </div>
          {chartLoading ? (
            <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
            </div>
          ) : totalConfidenceRecords > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={confidencePieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {confidencePieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
                <Legend 
                  wrapperStyle={{ fontSize: '12px' }}
                  formatter={(value, entry) => {
                    const item = confidencePieData.find(d => d.name === value)
                    return (
                      <span style={{ color: item?.color || '#000' }}>
                        {value} ({item?.value || 0})
                      </span>
                    )
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
              <div className="text-center">
                  <FaChartPie className="text-4xl text-gray-400 mx-auto mb-2" />
                <p className="text-sm text-gray-500">No data available</p>
              </div>
            </div>
          )}
        </motion.div>
      </motion.div>

      {/* Characters Over Time Chart - Full Width */}
      {characterChartData.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-white rounded-lg p-4 sm:p-6 shadow-sm border border-gray-200"
        >
          <div className="flex items-center space-x-2 sm:space-x-3 mb-4 sm:mb-6">
            <div className="p-2 sm:p-3 rounded-xl bg-purple-100">
              <FaFont className="text-xl sm:text-2xl text-purple-600" />
            </div>
            <div>
              <h3 className="text-lg sm:text-xl font-bold text-gray-800">Characters Recognized Over Time</h3>
              <p className="text-xs sm:text-sm text-gray-500">Total characters recognized per day (last 7 days)</p>
            </div>
          </div>
          {chartLoading ? (
            <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={characterChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis 
                  dataKey="date" 
                  stroke="#6b7280"
                  style={{ fontSize: '12px' }}
                />
                <YAxis 
                  stroke="#6b7280"
                  style={{ fontSize: '12px' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#fff', 
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar 
                  dataKey="characters" 
                  fill="#8b5cf6" 
                  radius={[8, 8, 0, 0]}
                  name="Characters Recognized"
                />
              </BarChart>
            </ResponsiveContainer>
          )}
        </motion.div>
      )}
    </div>
  )
}

export default AdminDashboard
