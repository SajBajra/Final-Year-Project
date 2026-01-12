import { useState, useEffect } from 'react'
import { FaSearch, FaFilter, FaDownload, FaSort, FaHistory, FaImage, FaTimes } from 'react-icons/fa'
import { getOCRHistory, exportOCRHistory } from '../../services/adminService'

const AdminOCRHistory = () => {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const [totalPages, setTotalPages] = useState(0)
  const [total, setTotal] = useState(0)
  
  // Filter states
  const [search, setSearch] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [sortBy, setSortBy] = useState('timestamp')
  const [sortOrder, setSortOrder] = useState('desc')
  const [showFilters, setShowFilters] = useState(false)
  const [exporting, setExporting] = useState(false)
  
  // Modal states
  const [imageModal, setImageModal] = useState({ isOpen: false, imagePath: null, filename: null })

  useEffect(() => {
    loadHistory()
  }, [page, search, startDate, endDate, sortBy, sortOrder])

  const buildFilters = () => {
    const filters = {}
    if (search.trim()) filters.search = search.trim()
    if (startDate) filters.startDate = new Date(startDate).toISOString()
    if (endDate) {
      const end = new Date(endDate)
      end.setHours(23, 59, 59, 999)
      filters.endDate = end.toISOString()
    }
    if (sortBy) filters.sortBy = sortBy
    if (sortOrder) filters.sortOrder = sortOrder
    return filters
  }

  const loadHistory = async () => {
    try {
      setLoading(true)
      const filters = buildFilters()
      const response = await getOCRHistory(page, 10, filters)
      if (response.success) {
        setHistory(response.data.data || [])
        setTotalPages(response.data.totalPages || 0)
        setTotal(response.data.total || 0)
      } else {
        console.error('Failed to load history:', response.message)
        setHistory([])
        setTotalPages(0)
        setTotal(0)
      }
    } catch (error) {
      console.error('Error loading history:', error)
      if (error.response) {
        console.error('Response status:', error.response.status)
        console.error('Response data:', error.response.data)
        if (error.response.status === 401 || error.response.status === 403) {
          alert('Authentication required. Please log in again.')
        }
      }
      setHistory([])
      setTotalPages(0)
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }

  const handleExport = async () => {
    try {
      setExporting(true)
      const filters = buildFilters()
      await exportOCRHistory(filters)
    } catch (error) {
      console.error('Error exporting history:', error)
      alert('Failed to export history')
    } finally {
      setExporting(false)
    }
  }

  const clearFilters = () => {
    setSearch('')
    setStartDate('')
    setEndDate('')
    setSortBy('timestamp')
    setSortOrder('desc')
    setPage(0)
  }

  const handleImageClick = (imagePath, filename) => {
    if (imagePath) {
      setImageModal({ isOpen: true, imagePath, filename })
    }
  }

  const hasActiveFilters = search || startDate || endDate

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
          <FaHistory className="text-primary-600" />
          OCR History
        </h2>
        <div className="flex items-center gap-4">
          <p className="text-sm text-gray-500">Total: {total} records</p>
          <button
            onClick={() => {
              if (showFilters) {
                // Clear filters when hiding
                clearFilters()
              }
              setShowFilters(!showFilters)
            }}
            className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2 text-sm font-semibold"
          >
            <FaFilter /> {showFilters ? 'Hide' : 'Show'} Filters
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 space-y-4">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search recognized text..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value)
                setPage(0)
              }}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-600 focus:border-transparent"
            />
          </div>
          <button
            onClick={handleExport}
            disabled={exporting}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 flex items-center gap-2"
          >
            <FaDownload /> {exporting ? 'Exporting...' : 'Export CSV'}
          </button>
        </div>

        {showFilters && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-200">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => {
                  setStartDate(e.target.value)
                  setPage(0)
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-600"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => {
                  setEndDate(e.target.value)
                  setPage(0)
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-600"
              />
            </div>
            <div className="flex items-end gap-2">
              <button
                onClick={clearFilters}
                className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg text-sm font-semibold"
              >
                Clear
              </button>
            </div>
          </div>
        )}

        {/* Sort Controls */}
        <div className="flex items-center gap-4 pt-2 border-t border-gray-200">
          <div className="flex items-center gap-2">
            <FaSort className="text-gray-400" />
            <span className="text-sm font-medium text-gray-700">Sort by:</span>
          </div>
          <select
            value={sortBy}
            onChange={(e) => {
              setSortBy(e.target.value)
              setPage(0)
            }}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-600"
          >
            <option value="timestamp">Timestamp</option>
            <option value="characterCount">Character Count</option>
          </select>
          <select
            value={sortOrder}
            onChange={(e) => {
              setSortOrder(e.target.value)
              setPage(0)
            }}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-600"
          >
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                SN
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Image
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Filename
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                User
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Recognized Text
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Characters
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Timestamp
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {history.length === 0 ? (
              <tr>
                <td colSpan="8" className="px-6 py-4 text-center text-gray-500">
                  {hasActiveFilters ? 'No records found matching filters' : 'No OCR history found'}
                </td>
              </tr>
            ) : (
              history.map((item) => (
                <tr key={item.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {item.id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {item.imagePath ? (
                      <button
                        onClick={() => handleImageClick(item.imagePath, item.imageFilename)}
                        className="w-16 h-16 border-2 border-gray-300 rounded-lg overflow-hidden hover:border-primary-500 transition-colors flex items-center justify-center bg-gray-50 group relative"
                        title="Click to view full image"
                      >
                        <img
                          src={`http://localhost:8080/api/images?path=${encodeURIComponent(item.imagePath)}`}
                          alt="OCR Image"
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            e.target.style.display = 'none'
                            e.target.parentElement.innerHTML = '<span class="text-gray-400 text-xs">No Image</span>'
                          }}
                        />
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                          <FaImage className="text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                        </div>
                      </button>
                    ) : (
                      <div className="w-16 h-16 border-2 border-gray-200 rounded-lg flex items-center justify-center bg-gray-100">
                        <span className="text-gray-400 text-xs">No Image</span>
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {item.imageFilename}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {item.isRegistered ? (
                      <div className="flex flex-col">
                        <span className="text-sm font-semibold text-gray-900 flex items-center gap-1">
                          {item.username || 'Unknown User'}
                          {item.userRole === 'ADMIN' && (
                            <span className="inline-block px-2 py-0.5 bg-red-100 text-red-800 text-xs font-bold rounded">
                              ADMIN
                            </span>
                          )}
                          {item.userRole === 'USER' && (
                            <span className="inline-block px-2 py-0.5 bg-blue-100 text-blue-800 text-xs font-bold rounded">
                              USER
                            </span>
                          )}
                        </span>
                        <span className="text-xs text-gray-500">{item.userEmail || 'No email'}</span>
                      </div>
                    ) : (
                      <div className="flex flex-col">
                        <span className="text-sm font-semibold text-gray-500 flex items-center gap-1">
                          Guest User
                          <span className="inline-block px-2 py-0.5 bg-gray-100 text-gray-600 text-xs font-bold rounded">
                            GUEST
                          </span>
                        </span>
                        <span className="text-xs text-gray-400">Unregistered</span>
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900 max-w-xs truncate" title={item.recognizedText}>
                    {item.recognizedText}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {item.characterCount}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'N/A'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <button
            onClick={() => setPage(page - 1)}
            disabled={page === 0}
            className="px-4 py-2 bg-white border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Previous
          </button>
          <span className="text-sm text-gray-500">
            Page {page + 1} of {totalPages} ({total} total records)
          </span>
          <button
            onClick={() => setPage(page + 1)}
            disabled={page >= totalPages - 1}
            className="px-4 py-2 bg-white border border-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Next
          </button>
        </div>
      )}

      {/* Image Viewing Modal */}
      {imageModal.isOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={() => setImageModal({ isOpen: false, imagePath: null, filename: null })}
        >
          <div
            className="relative bg-white rounded-lg shadow-2xl max-w-5xl w-full max-h-[90vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white rounded-t-lg flex-shrink-0">
              <div className="min-w-0 flex-1 mr-4">
                <h3 className="text-lg font-semibold text-gray-900 truncate">OCR Image</h3>
                <p className="text-sm text-gray-600 mt-1 truncate">{imageModal.filename || 'Image'}</p>
              </div>
              <button
                onClick={() => setImageModal({ isOpen: false, imagePath: null, filename: null })}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors flex-shrink-0"
              >
                <FaTimes className="text-xl text-gray-600" />
              </button>
            </div>
            <div className="p-6 overflow-auto flex-1">
              <img
                src={`http://localhost:8080/api/images?path=${encodeURIComponent(imageModal.imagePath)}`}
                alt={imageModal.filename || 'OCR Image'}
                className="w-full h-auto rounded-lg shadow-lg"
                onError={(e) => {
                  e.target.style.display = 'none'
                  e.target.parentElement.innerHTML = '<div class="text-center text-red-600 py-8"><p>Failed to load image</p><p class="text-sm text-gray-500 mt-2">' + imageModal.imagePath + '</p></div>'
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AdminOCRHistory
