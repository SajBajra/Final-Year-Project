const Header = () => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4 max-w-6xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="text-3xl">📜</div>
            <div>
              <h1 className="text-2xl font-bold text-gradient">Lipika</h1>
              <p className="text-xs text-gray-500">Ranjana OCR System</p>
            </div>
          </div>
          <nav className="flex items-center space-x-6">
            <a href="#about" className="text-gray-600 hover:text-blue-600 font-medium transition-colors">
              About
            </a>
            <a href="#features" className="text-gray-600 hover:text-blue-600 font-medium transition-colors">
              Features
            </a>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-blue-600 font-medium transition-colors"
            >
              GitHub
            </a>
          </nav>
        </div>
      </div>
    </header>
  )
}

export default Header

