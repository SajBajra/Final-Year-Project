import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { motion } from 'framer-motion'
import { AuthProvider } from './context/AuthContext'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import Features from './pages/Features'
import About from './pages/About'
import Pricing from './pages/Pricing'
import Contact from './pages/Contact'
import Login from './pages/Login'
import Register from './pages/Register'
import ForgotPassword from './pages/ForgotPassword'
import ResetPassword from './pages/ResetPassword'
import UserProfile from './pages/UserProfile'
import ChangePassword from './pages/ChangePassword'
import Payment from './pages/Payment'
import PaymentSuccess from './pages/PaymentSuccess'
import PaymentFailure from './pages/PaymentFailure'
import AdminLayout from './components/AdminLayout'
import AdminDashboard from './pages/admin/AdminDashboard'
import AdminOCRHistory from './pages/admin/AdminOCRHistory'
import AdminAnalytics from './pages/admin/AdminAnalytics'
import AdminCharacterStats from './pages/admin/AdminCharacterStats'
import AdminUserManagement from './pages/admin/AdminUserManagement'
import AdminContacts from './pages/admin/AdminContacts'
import AdminSettings from './pages/admin/AdminSettings'
import ProtectedRoute from './components/ProtectedRoute'
import Preloader from './components/Preloader'
import ScrollToTop from './components/ScrollToTop'
import { ROUTES } from './config/constants'
import './index.css'

function App() {
  return (
    <AuthProvider>
      <Router>
        <ScrollToTop />
        <Preloader />
        <Routes>
        {/* Public Routes */}
        <Route path="/" element={
          <div className="min-h-screen bg-primary-50 flex flex-col">
            <Header />
            <div className="flex-grow">
              <Home />
            </div>
            <Footer />
          </div>
        } />
        <Route path="/features" element={
          <div className="min-h-screen bg-primary-50 flex flex-col">
            <Header />
            <div className="flex-grow">
              <Features />
            </div>
            <Footer />
          </div>
        } />
        <Route path="/about" element={
          <div className="min-h-screen bg-primary-50 flex flex-col">
            <Header />
            <div className="flex-grow">
              <About />
            </div>
            <Footer />
          </div>
        } />
        
        <Route path="/pricing" element={
          <div className="min-h-screen bg-primary-50 flex flex-col">
            <Header />
            <div className="flex-grow">
              <Pricing />
            </div>
            <Footer />
          </div>
        } />
        
        <Route path="/contact" element={
          <div className="min-h-screen bg-primary-50 flex flex-col">
            <Header />
            <div className="flex-grow">
              <Contact />
            </div>
            <Footer />
          </div>
        } />
        
        {/* Auth Routes */}
        <Route path="/register" element={
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Register />
          </motion.div>
        } />
        
        <Route path="/login" element={
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <Login />
          </motion.div>
        } />
        
        <Route path="/forgot-password" element={
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <ForgotPassword />
          </motion.div>
        } />
        
        <Route path="/reset-password" element={
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <ResetPassword />
          </motion.div>
        } />
        
        <Route path="/profile" element={
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <UserProfile />
          </motion.div>
        } />
        
        <Route path="/change-password" element={
          <ProtectedRoute>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <ChangePassword />
            </motion.div>
          </ProtectedRoute>
        } />
        
        {/* Payment Routes */}
        <Route path="/payment" element={
          <ProtectedRoute>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Payment />
            </motion.div>
          </ProtectedRoute>
        } />
        
        <Route path="/payment/success" element={
          <ProtectedRoute>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <PaymentSuccess />
            </motion.div>
          </ProtectedRoute>
        } />
        
        <Route path="/payment/failure" element={
          <ProtectedRoute>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <PaymentFailure />
            </motion.div>
          </ProtectedRoute>
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
          <Route path="users" element={<AdminUserManagement />} />
          <Route path="contacts" element={<AdminContacts />} />
          <Route path="settings" element={<AdminSettings />} />
        </Route>
        </Routes>
      </Router>
    </AuthProvider>
  )
}

export default App

