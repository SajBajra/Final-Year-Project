import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  FaEnvelope, 
  FaEnvelopeOpen,
  FaUser,
  FaClock
} from 'react-icons/fa';
import { getAllContacts, markContactAsRead } from '../../services/adminService';
import ConfirmationModal from '../../components/ConfirmationModal';

const AdminContacts = () => {
  const [contacts, setContacts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedContact, setSelectedContact] = useState(null);
  const [showMessageModal, setShowMessageModal] = useState(false);

  useEffect(() => {
    loadContacts();
  }, []);

  const loadContacts = async () => {
    try {
      setLoading(true);
      const response = await getAllContacts();
      if (response.success) {
        setContacts(response.data);
      } else {
        setError(response.message || 'Failed to load contacts');
      }
    } catch (err) {
      setError('Failed to load contacts');
      console.error('Error loading contacts:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleMarkAsRead = async (contactId) => {
    try {
      const response = await markContactAsRead(contactId);
      if (response.success) {
        loadContacts();
      }
    } catch (err) {
      console.error('Error marking contact as read:', err);
    }
  };

  const handleViewMessage = (contact) => {
    setSelectedContact(contact);
    setShowMessageModal(true);
    if (!contact.read) {
      handleMarkAsRead(contact.id);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const unreadCount = contacts.filter(c => !c.read).length;

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Contact Submissions</h1>
          <p className="text-gray-600 mt-1">Manage and respond to user inquiries</p>
        </div>
        {unreadCount > 0 && (
          <div className="bg-primary-100 text-primary-700 px-4 py-2 rounded-lg font-semibold">
            {unreadCount} Unread
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
          {error}
        </div>
      )}

      {/* Contacts List */}
      {contacts.length === 0 ? (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <FaEnvelope className="text-6xl text-gray-300 mx-auto mb-4" />
          <p className="text-gray-600 text-lg">No contact submissions yet</p>
        </div>
      ) : (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Name
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Email
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Subject
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {contacts.map((contact) => (
                  <motion.tr
                    key={contact.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className={`hover:bg-gray-50 transition-colors ${!contact.read ? 'bg-primary-50' : ''}`}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      {contact.read ? (
                        <FaEnvelopeOpen className="text-gray-400" />
                      ) : (
                        <FaEnvelope className="text-primary-600" />
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <FaUser className="text-gray-400 mr-2" />
                        <span className={`font-medium ${!contact.read ? 'text-gray-900 font-semibold' : 'text-gray-700'}`}>
                          {contact.name}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      {contact.email}
                    </td>
                    <td className="px-6 py-4 max-w-xs truncate text-sm text-gray-700">
                      {contact.subject}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      <div className="flex items-center">
                        <FaClock className="mr-2 text-gray-400" />
                        {formatDate(contact.submittedAt)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                      <button
                        onClick={() => handleViewMessage(contact)}
                        className="text-primary-600 hover:text-primary-900 transition-colors"
                      >
                        View
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* View Message Modal */}
      <ConfirmationModal
        isOpen={showMessageModal && selectedContact !== null}
        onClose={() => {
          setShowMessageModal(false);
          setSelectedContact(null);
        }}
        maxWidth="max-w-2xl"
        showButtons={false}
      >
        <div className="mt-6">
          <h3 className="text-2xl font-bold text-gray-900 mb-4">
            {selectedContact?.subject}
          </h3>
          
          <div className="space-y-4 mb-6">
            <div className="flex items-center gap-3 text-gray-600">
              <FaUser className="text-primary-600" />
              <span className="font-medium">{selectedContact?.name}</span>
            </div>
            <div className="flex items-center gap-3 text-gray-600">
              <FaEnvelope className="text-primary-600" />
              <a href={`mailto:${selectedContact?.email}`} className="text-primary-600 hover:underline">
                {selectedContact?.email}
              </a>
            </div>
            <div className="flex items-center gap-3 text-gray-600">
              <FaClock className="text-primary-600" />
              <span>{formatDate(selectedContact?.submittedAt)}</span>
            </div>
          </div>

          <div className="border-t border-gray-200 pt-4">
            <h4 className="font-semibold text-gray-900 mb-2">Message:</h4>
            <p className="text-gray-700 whitespace-pre-wrap">{selectedContact?.message}</p>
          </div>

          <div className="mt-6 flex justify-end">
            <button
              onClick={() => {
                setShowMessageModal(false);
                setSelectedContact(null);
              }}
              className="btn-primary px-6 py-2"
            >
              Close
            </button>
          </div>
        </div>
      </ConfirmationModal>

    </div>
  );
};

export default AdminContacts;
