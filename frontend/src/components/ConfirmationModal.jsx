import { motion, AnimatePresence } from 'framer-motion';
import { FaTimes } from 'react-icons/fa';

/**
 * Reusable Modal Component
 * 
 * @param {boolean} isOpen - Controls modal visibility
 * @param {function} onClose - Function to call when closing the modal
 * @param {string} title - Modal title
 * @param {string|React.Node} message - Modal message or custom content
 * @param {React.Node} icon - Icon to display at the top (optional)
 * @param {string} iconBgColor - Background color for icon (e.g., 'bg-red-100')
 * @param {string} confirmText - Text for confirm button
 * @param {string} confirmIcon - Icon component for confirm button (optional)
 * @param {string} confirmClassName - Classes for confirm button
 * @param {function} onConfirm - Function to call on confirm
 * @param {string} cancelText - Text for cancel button (default: 'Cancel')
 * @param {boolean} showButtons - Whether to show action buttons (default: true)
 * @param {string} maxWidth - Max width class (default: 'max-w-md')
 * @param {React.Node} children - Custom content to replace default message
 */
const ConfirmationModal = ({
  isOpen,
  onClose,
  title,
  message,
  icon,
  iconBgColor = 'bg-red-100',
  confirmText,
  confirmIcon: ConfirmIcon,
  confirmClassName = 'bg-red-600 hover:bg-red-700',
  onConfirm,
  cancelText = 'Cancel',
  showButtons = true,
  maxWidth = 'max-w-md',
  children
}) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-[100]"
          />
          
          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-[101] flex items-center justify-center p-4 pointer-events-none"
          >
            <div 
              className={`bg-white rounded-2xl shadow-2xl ${maxWidth} w-full p-6 pointer-events-auto`}
              onClick={(e) => e.stopPropagation()}
            >
              {/* Custom Content or Default Layout */}
              {children ? (
                children
              ) : (
                <>
                  {/* Icon */}
                  {icon && (
                    <div className="flex justify-center mb-4">
                      <div className={`w-16 h-16 rounded-full ${iconBgColor} flex items-center justify-center`}>
                        {icon}
                      </div>
                    </div>
                  )}

                  {/* Title */}
                  {title && (
                    <h3 className="text-2xl font-bold text-gray-900 text-center mb-2">
                      {title}
                    </h3>
                  )}

                  {/* Message */}
                  {message && (
                    <div className="text-gray-600 text-center mb-6">
                      {typeof message === 'string' ? <p>{message}</p> : message}
                    </div>
                  )}

                  {/* Buttons */}
                  {showButtons && (
                    <div className="flex gap-3">
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={onClose}
                        className="flex-1 px-4 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded-lg transition-colors"
                      >
                        {cancelText}
                      </motion.button>
                      {onConfirm && (
                        <motion.button
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          onClick={onConfirm}
                          className={`flex-1 px-4 py-3 text-white font-semibold rounded-lg transition-colors flex items-center justify-center gap-2 ${confirmClassName}`}
                        >
                          {ConfirmIcon && <ConfirmIcon />}
                          {confirmText}
                        </motion.button>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default ConfirmationModal;
