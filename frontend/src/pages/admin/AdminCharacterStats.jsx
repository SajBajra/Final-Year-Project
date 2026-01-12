import { useState, useEffect } from 'react'
import { FaFont, FaChartBar, FaPercent, FaInfoCircle } from 'react-icons/fa'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'
import { getCharacterStatistics } from '../../services/adminService'
import { UI_CONFIG } from '../../config/constants'

const AdminCharacterStats = () => {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [currentPage, setCurrentPage] = useState(0)
  const [itemsPerPage] = useState(10) // 10 characters per page

  // Use primary and secondary colors with variations
  const COLORS = [
    UI_CONFIG.PRIMARY_COLOR, 
    UI_CONFIG.SECONDARY_COLOR, 
    UI_CONFIG.PRIMARY_LIGHT, 
    UI_CONFIG.SECONDARY_LIGHT, 
    UI_CONFIG.PRIMARY_DARK, 
    UI_CONFIG.SECONDARY_DARK,
    '#6366f1', // Additional shade for variety
    '#14b8a6' // Additional shade for variety
  ]

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
    frequency: char.frequency
  }))

  // Pagination calculations
  const totalPages = Math.ceil(topCharacters.length / itemsPerPage)
  const startIndex = currentPage * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const paginatedCharacters = topCharacters.slice(startIndex, endIndex)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-black flex items-center gap-3">
          <FaFont className="text-primary-600" />
          Character Statistics
        </h2>
        <div className="flex items-center gap-2 text-sm text-black">
          <FaInfoCircle />
          <span>Analysis of recognized characters across all OCR requests</span>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-black font-semibold">Total Unique Characters</p>
              {loading ? (
                <div className="h-10 w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <p className="text-3xl font-bold text-black mt-2">
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
              <p className="text-sm text-black font-semibold">Top Characters Analyzed</p>
              {loading ? (
                <div className="h-10 w-20 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <>
                  <p className="text-3xl font-bold text-black mt-2">
                    {topCharacters.length}
                  </p>
                  <p className="text-xs text-black mt-1">Displaying top 20</p>
                </>
              )}
            </div>
            <FaChartBar className="text-4xl text-primary-600" />
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-black font-semibold">Most Frequent Character</p>
              {loading ? (
                <div className="h-10 w-32 bg-gray-200 animate-pulse-fast rounded mt-2 transition-fast"></div>
              ) : (
                <p className="text-2xl font-bold text-black mt-2">
                  {topCharacters.length > 0 ? (
                    <>
                      {topCharacters[0].character === ' ' ? 'Space' : topCharacters[0].character}
                      <span className="text-sm text-black ml-2">
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
          <h3 className="text-xl font-bold text-black mb-4">Top 20 Most Recognized Characters</h3>
          <div className="h-[400px] bg-gray-100 animate-pulse-fast rounded transition-fast"></div>
        </div>
      ) : chartData.length > 0 && (
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <h3 className="text-xl font-bold text-black mb-4">Top 20 Most Recognized Characters</h3>
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
              <YAxis label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="frequency" fill={UI_CONFIG.PRIMARY_COLOR} name="Recognition Frequency">
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
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
              <div key={i} className="h-12 bg-gray-100 animate-pulse-fast rounded transition-fast"></div>
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
                  <th className="px-6 py-3 text-left text-xs font-medium text-black uppercase tracking-wider">
                    Rank
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-black uppercase tracking-wider">
                    Character
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-black uppercase tracking-wider">
                    Unicode
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-black uppercase tracking-wider">
                    Frequency
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-black uppercase tracking-wider">
                    Percentage
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {paginatedCharacters.map((char, index) => {
                  const totalFrequency = topCharacters.reduce((sum, c) => sum + c.frequency, 0)
                  const percentage = totalFrequency > 0 
                    ? ((char.frequency / totalFrequency) * 100).toFixed(2) 
                    : 0
                  const actualIndex = startIndex + index
                  
                  return (
                    <tr key={actualIndex} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        #{actualIndex + 1}
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
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                        {percentage}%
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          
          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between bg-gray-50">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
                  disabled={currentPage === 0}
                  className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Previous
                </button>
                <button
                  onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
                  disabled={currentPage >= totalPages - 1}
                  className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Next
                </button>
              </div>
              <div className="text-sm text-gray-700">
                Page <span className="font-semibold">{currentPage + 1}</span> of <span className="font-semibold">{totalPages}</span>
                {' '}• Showing <span className="font-semibold">{startIndex + 1}</span> to <span className="font-semibold">{Math.min(endIndex, topCharacters.length)}</span> of <span className="font-semibold">{topCharacters.length}</span> characters
              </div>
            </div>
          )}
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

