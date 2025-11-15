import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import Features from './pages/Features'
import About from './pages/About'
import Login from './pages/Login'
import Register from './pages/Register'
import AdminLayout from './components/AdminLayout'
import AdminDashboard from './pages/admin/AdminDashboard'
import AdminOCRHistory from './pages/admin/AdminOCRHistory'
import AdminAnalytics from './pages/admin/AdminAnalytics'
import AdminCharacterStats from './pages/admin/AdminCharacterStats'
import AdminSettings from './pages/admin/AdminSettings'
import ProtectedRoute from './components/ProtectedRoute'
import { ROUTES } from './config/constants'
import './index.css'

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
        {/* Public Routes */}
        <Route path="/" element={
          <div className="min-h-screen bg-gray-50 flex flex-col">
            <Header />
            <div className="flex-grow">
              <Home />
            </div>
            <Footer />
          </div>
        } />
        <Route path="/features" element={
          <div className="min-h-screen bg-gray-50 flex flex-col">
            <Header />
            <div className="flex-grow">
              <Features />
            </div>
            <Footer />
          </div>
        } />
        <Route path="/about" element={
          <div className="min-h-screen bg-gray-50 flex flex-col">
            <Header />
            <div className="flex-grow">
              <About />
            </div>
            <Footer />
          </div>
        } />
        
        {/* Auth Routes with AnimatePresence for page transitions */}
        <Route path="/login" element={
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Login />
          </motion.div>
        } />
        <Route path="/register" element={
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Register />
          </motion.div>
        } />
        
        {/* Admin Routes */}
        <Route path={ROUTES.ADMIN} element={
          <ProtectedRoute requireAdmin={true}>
            <AdminLayout />
          </ProtectedRoute>
        }>
          <Route index element={<AdminDashboard />} />
          <Route path="dashboard" element={<AdminDashboard />} />
          <Route path="ocr-history" element={<AdminOCRHistory />} />
          <Route path="analytics" element={<AdminAnalytics />} />
          <Route path="characters" element={<AdminCharacterStats />} />
          <Route path="settings" element={<AdminSettings />} />
        </Route>
        </Routes>
      </Router>
    </AuthProvider>
  )
}

export default App

