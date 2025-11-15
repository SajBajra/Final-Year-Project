import { useState, useEffect } from 'react'
import { FaFont, FaChartBar, FaPercent, FaInfoCircle } from 'react-icons/fa'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'
import { getCharacterStatistics } from '../../services/adminService'

const AdminCharacterStats = () => {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#6366f1']

  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    try {
      setLoading(true)
      const response = await getCharacterStatistics()
      if (response.success) {
        setStats(response.data)
      }
    } catch (error) {
      console.error('Error loading character statistics:', error)
    } finally {
      setLoading(false)
    }
  }

  const topCharacters = stats?.topCharacters || []
  const chartData = topCharacters.map((char, index) => ({
    character: char.character === ' ' ? 'Space' : char.character,
    frequency: char.frequency,
    avgConfidence: (char.avgConfidence * 100).toFixed(1)
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-800">Character Statistics</h2>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <FaInfoCircle />
          <span>Analysis of recognized characters across all OCR requests</span>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Total Unique Characters</p>
              {loading ? (
                <div className="h-10 w-20 bg-gray-200 animate-pulse rounded mt-2"></div>
              ) : (
                <p className="text-3xl font-bold text-gray-800 mt-2">
                  {stats?.totalUniqueCharacters ?? 0}
                </p>
              )}
            </div>
            <FaFont className="text-4xl text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Top Characters Analyzed</p>
              {loading ? (
                <div className="h-10 w-20 bg-gray-200 animate-pulse rounded mt-2"></div>
              ) : (
                <>
                  <p className="text-3xl font-bold text-gray-800 mt-2">
                    {topCharacters.length}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">Displaying top 20</p>
                </>
              )}
            </div>
            <FaChartBar className="text-4xl text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500 font-semibold">Most Frequent Character</p>
              {loading ? (
                <div className="h-10 w-32 bg-gray-200 animate-pulse rounded mt-2"></div>
              ) : (
                <p className="text-2xl font-bold text-gray-800 mt-2">
                  {topCharacters.length > 0 ? (
                    <>
                      {topCharacters[0].character === ' ' ? 'Space' : topCharacters[0].character}
                      <span className="text-sm text-gray-500 ml-2">
                        ({topCharacters[0].frequency} times)
                      </span>
                    </>
                  ) : 'N/A'}
                </p>
              )}
            </div>
            <FaPercent className="text-4xl text-primary-600" />
          </div>
        </div>
      </div>

      {/* Top Characters Chart */}
      {loading ? (
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Top 20 Most Recognized Characters</h3>
          <div className="h-[400px] bg-gray-100 animate-pulse rounded"></div>
        </div>
      ) : chartData.length > 0 && (
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Top 20 Most Recognized Characters</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="character" 
                angle={-45}
                textAnchor="end"
                height={100}
                interval={0}
              />
              <YAxis yAxisId="left" label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} />
              <YAxis 
                yAxisId="right" 
                orientation="right"
                domain={[0, 100]}
                label={{ value: 'Avg Confidence (%)', angle: 90, position: 'insideRight' }}
              />
              <Tooltip />
              <Legend />
              <Bar yAxisId="left" dataKey="frequency" fill="#2563eb" name="Recognition Frequency">
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
              <Bar 
                yAxisId="right" 
                dataKey="avgConfidence" 
                fill="#10b981" 
                name="Avg Confidence (%)"
                opacity={0.7}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Top Characters Table */}
      {loading ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <h3 className="text-xl font-bold text-gray-800 p-6 border-b border-gray-200">
            Character Details
          </h3>
          <div className="p-6 space-y-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-12 bg-gray-100 animate-pulse rounded"></div>
            ))}
          </div>
        </div>
      ) : topCharacters.length > 0 ? (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
          <h3 className="text-xl font-bold text-gray-800 p-6 border-b border-gray-200">
            Character Details
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Rank
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Character
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Unicode
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Frequency
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Percentage
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {topCharacters.map((char, index) => {
                  const totalFrequency = topCharacters.reduce((sum, c) => sum + c.frequency, 0)
                  const percentage = totalFrequency > 0 
                    ? ((char.frequency / totalFrequency) * 100).toFixed(2) 
                    : 0
                  
                  return (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        #{index + 1}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className="text-2xl font-bold text-gray-900">
                          {char.character === ' ' ? '␣' : char.character}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-mono">
                        U+{char.character.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-semibold">
                        {char.frequency.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className={`font-semibold ${
                          char.avgConfidence >= 0.8 ? 'text-green-600' :
                          char.avgConfidence >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {(char.avgConfidence * 100).toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                        {percentage}%
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-lg p-12 shadow-sm border border-gray-200 text-center">
          <FaInfoCircle className="text-5xl text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 text-lg">No character statistics available yet</p>
          <p className="text-gray-400 text-sm mt-2">Character statistics will appear after OCR requests are processed</p>
        </div>
      )}
    </div>
  )
}

export default AdminCharacterStats

