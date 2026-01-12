import { useState, useEffect } from 'react'
import { FaChartLine, FaChartBar, FaChartPie, FaDownload } from 'react-icons/fa'
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { getAnalytics } from '../../services/adminService'
import { UI_CONFIG } from '../../config/constants'

const AdminAnalytics = () => {
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [period, setPeriod] = useState('daily')
  const [days, setDays] = useState(30)

  // Use primary and secondary colors with variations
  const COLORS = [
    UI_CONFIG.PRIMARY_COLOR, 
    UI_CONFIG.SECONDARY_COLOR, 
    UI_CONFIG.PRIMARY_LIGHT, 
    UI_CONFIG.SECONDARY_LIGHT, 
    UI_CONFIG.PRIMARY_DARK
  ]

  useEffect(() => {
    loadAnalytics()
  }, [period, days])

  const loadAnalytics = async () => {
    try {
      setLoading(true)
      const response = await getAnalytics(period, days)
      if (response.success) {
        setAnalytics(response.data)
      }
    } catch (error) {
      console.error('Error loading analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  const formatTimeSeriesData = () => {
    if (!analytics || !analytics.timeSeries) return []
    
    const timeSeries = analytics.timeSeries
    const characterSeries = analytics.characterSeries || {}
    
    return Object.keys(timeSeries).map(key => ({
      date: key,
      requests: timeSeries[key] || 0,
      characters: characterSeries[key] || 0
    }))
  }

  const formatTextLengthDistribution = () => {
    if (!analytics || !analytics.textLengthDistribution) return []
    
    const dist = analytics.textLengthDistribution
    return Object.keys(dist).map(key => ({
      name: key,
      value: dist[key]
    }))
  }

  const chartData = formatTimeSeriesData()
  const distributionData = formatTextLengthDistribution()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
          <FaChartBar className="text-primary-600" />
          Analytics Dashboard
        </h2>
        <div className="flex items-center gap-4">
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-600"
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
          <select
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value))}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-600"
          >
            <option value="7">Last 7 days</option>
            <option value="14">Last 14 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
            <option value="365">Last year</option>
          </select>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Total Records</p>
              {loading ? (
                <div className="h-10 w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <p className="text-3xl font-bold text-gray-800 mt-2">
                  {analytics?.totalRecords ?? 0}
                </p>
              )}
            </div>
            <FaChartLine className="text-4xl text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Period</p>
              {loading ? (
                <div className="h-10 w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <>
                  <p className="text-2xl font-bold text-gray-800 mt-2 capitalize">
                    {analytics?.period || period}
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    {analytics?.days || days} days
                  </p>
                </>
              )}
            </div>
            <FaChartBar className="text-4xl text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Data Points</p>
              {loading ? (
                <div className="h-10 w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <p className="text-3xl font-bold text-gray-800 mt-2">
                  {chartData.length}
                </p>
              )}
            </div>
            <FaChartPie className="text-4xl text-primary-600" />
          </div>
        </div>
      </div>

      {/* OCR Requests Over Time */}
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
        <h3 className="text-xl font-bold text-gray-800 mb-4">OCR Requests Over Time</h3>
        {loading ? (
          <div className="h-[300px] bg-gray-100 animate-pulse-fast rounded transition-fast"></div>
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="requests" 
                stroke={UI_CONFIG.PRIMARY_COLOR} 
                strokeWidth={2}
                name="OCR Requests"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center text-gray-500 py-8">No data available for selected period</div>
        )}
      </div>

      {/* Characters Recognized Over Time */}
      <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Characters Recognized Over Time</h3>
        {loading ? (
          <div className="h-[300px] bg-gray-100 animate-pulse-fast rounded transition-fast"></div>
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar 
                dataKey="characters" 
                fill={UI_CONFIG.SECONDARY_COLOR}
                name="Characters Recognized"
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center text-gray-500 py-8">No data available for selected period</div>
        )}
      </div>

      {/* Text Length Distribution */}
      {loading ? (
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Text Length Distribution</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="h-[300px] bg-gray-100 animate-pulse-fast rounded transition-fast"></div>
            <div className="space-y-4">
              <div className="h-6 bg-gray-200 animate-pulse-fast rounded transition-fast"></div>
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-12 bg-gray-100 animate-pulse-fast rounded transition-fast"></div>
              ))}
            </div>
          </div>
        </div>
      ) : distributionData.length > 0 && (
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Text Length Distribution</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={distributionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill={UI_CONFIG.PRIMARY_COLOR}
                  dataKey="value"
                >
                  {distributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-4">
              <h4 className="font-semibold text-gray-700">Distribution Breakdown</h4>
              {distributionData.map((item, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-4 h-4 rounded"
                      style={{ backgroundColor: COLORS[index % COLORS.length] }}
                    />
                    <span className="font-medium text-gray-700">{item.name}</span>
                  </div>
                  <span className="font-bold text-gray-900">{item.value} records</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AdminAnalytics

