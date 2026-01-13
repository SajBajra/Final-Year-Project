import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaUsers, FaSearch, FaTrash, FaCrown, FaStar, FaUser, FaExclamationTriangle, FaSignOutAlt } from 'react-icons/fa';
import { getAllUsers, deleteUser } from '../../services/adminService';
import ConfirmationModal from '../../components/ConfirmationModal';

const AdminUserManagement = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('All');
  const [selectedUser, setSelectedUser] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  useEffect(() => {
    loadUsers();
  }, []);

  const loadUsers = async () => {
    try {
      setLoading(true);
      const response = await getAllUsers();
      if (response.success) {
        setUsers(response.data);
      }
    } catch (error) {
      console.error('Error loading users:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteUser = async (userId) => {
    try {
      const response = await deleteUser(userId);
      if (response.success) {
        await loadUsers();
        setShowDeleteModal(false);
        setSelectedUser(null);
        alert('User deleted successfully!');
      } else {
        alert(response.message || 'Failed to delete user');
      }
    } catch (error) {
      console.error('Error deleting user:', error);
      const errorMessage = error.response?.data?.message || error.message || 'Failed to delete user. Please try again.';
      alert(errorMessage);
    }
  };

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterType === 'All' || user.accountType === filterType;
    return matchesSearch && matchesFilter;
  });

  const getAccountTypeBadge = (accountType, planType) => {
    switch (accountType) {
      case 'Admin':
        return (
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-primary-600 text-white">
            <FaCrown className="mr-1" /> Admin
          </span>
        );
      case 'Paid':
        const planLabel = planType ? ` (${planType.charAt(0).toUpperCase() + planType.slice(1)})` : '';
        return (
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-gradient-to-r from-yellow-400 to-yellow-500 text-gray-900">
            <FaStar className="mr-1" /> Paid{planLabel}
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-gray-200 text-gray-700">
            <FaUser className="mr-1" /> Free
          </span>
        );
    }
  };

  const stats = {
    total: users.length,
    free: users.filter(u => u.accountType === 'Free').length,
    paid: users.filter(u => u.accountType === 'Paid').length,
    admin: users.filter(u => u.accountType === 'Admin').length
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-black flex items-center gap-3">
          <FaUsers className="text-primary-600" />
          User Management
        </h2>
        <button
          onClick={loadUsers}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          Refresh
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <p className="text-sm text-gray-600 font-semibold">Total Users</p>
          <p className="text-3xl font-bold text-gray-900 mt-2">{stats.total}</p>
        </div>
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <p className="text-sm text-gray-600 font-semibold">Free Users</p>
          <p className="text-3xl font-bold text-gray-700 mt-2">{stats.free}</p>
        </div>
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <p className="text-sm text-gray-600 font-semibold">Paid Users</p>
          <p className="text-3xl font-bold text-yellow-600 mt-2">{stats.paid}</p>
        </div>
        <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
          <p className="text-sm text-gray-600 font-semibold">Admins</p>
          <p className="text-3xl font-bold text-purple-600 mt-2">{stats.admin}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg p-4 shadow-sm border border-gray-200">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search by username or email..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>

          {/* Filter by type */}
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="All">All Users</option>
            <option value="Free">Free</option>
            <option value="Paid">Paid</option>
            <option value="Admin">Admin</option>
          </select>
        </div>
      </div>

      {/* Users Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Email</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Account Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Usage</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Joined</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {loading ? (
                <tr>
                  <td colSpan="6" className="px-6 py-12 text-center text-gray-500">
                    Loading users...
                  </td>
                </tr>
              ) : filteredUsers.length === 0 ? (
                <tr>
                  <td colSpan="6" className="px-6 py-12 text-center text-gray-500">
                    No users found
                  </td>
                </tr>
              ) : (
                filteredUsers.map((user) => (
                  <tr key={user.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="font-medium text-gray-900">{user.username}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-600">{user.email}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {getAccountTypeBadge(user.accountType, user.planType)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        {user.accountType === 'Admin' ? (
                          <span className="text-purple-600 font-semibold">Unlimited</span>
                        ) : (
                          <>
                            {user.usageCount} / {user.accountType === 'Paid' ? '∞' : user.usageLimit}
                          </>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-600">
                        {new Date(user.createdAt).toLocaleDateString()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex items-center gap-2">
                        {/* Show delete button only for Free and Paid users (not Admin) */}
                        {user.accountType !== 'Admin' ? (
                          <button
                            onClick={() => {
                              setSelectedUser(user);
                              setShowDeleteModal(true);
                            }}
                            className="text-red-600 hover:text-red-900"
                            title="Delete User"
                          >
                            <FaTrash />
                          </button>
                        ) : (
                          <span className="text-gray-400 text-xs">No actions</span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={showDeleteModal && selectedUser !== null}
        onClose={() => {
          setShowDeleteModal(false);
          setSelectedUser(null);
        }}
        title="Delete User"
        message={
          selectedUser && (
            <>
              Are you sure you want to delete <strong className="text-gray-900">{selectedUser.username}</strong>? This action cannot be undone and will remove all user data including OCR history and payment records.
            </>
          )
        }
        icon={<FaExclamationTriangle className="text-3xl text-red-600" />}
        iconBgColor="bg-red-100"
        confirmText="Delete"
        confirmIcon={FaTrash}
        confirmClassName="bg-red-600 hover:bg-red-700"
        onConfirm={() => selectedUser && handleDeleteUser(selectedUser.id)}
      />
    </div>
  );
};

export default AdminUserManagement;
