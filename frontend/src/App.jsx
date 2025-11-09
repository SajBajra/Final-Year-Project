import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import Home from './pages/Home'
import Features from './pages/Features'
import About from './pages/About'
import AdminLayout from './components/AdminLayout'
import AdminDashboard from './pages/admin/AdminDashboard'
import AdminOCRHistory from './pages/admin/AdminOCRHistory'
import AdminSettings from './pages/admin/AdminSettings'
import { ROUTES } from './config/constants'
import './index.css'

function App() {
  return (
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
        
        {/* Admin Routes */}
        <Route path={ROUTES.ADMIN} element={<AdminLayout />}>
          <Route index element={<AdminDashboard />} />
          <Route path="dashboard" element={<AdminDashboard />} />
          <Route path="ocr-history" element={<AdminOCRHistory />} />
          <Route path="settings" element={<AdminSettings />} />
        </Route>
      </Routes>
    </Router>
  )
}

export default App

