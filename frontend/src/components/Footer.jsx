const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white mt-20">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          <div>
            <h3 className="text-2xl font-bold mb-4">Lipika</h3>
            <p className="text-gray-400 text-sm">
              Advanced OCR system for Ranjana script with AR support, powered by deep learning.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Technology</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>• Python + PyTorch</li>
              <li>• React + TailwindCSS</li>
              <li>• CRNN Architecture</li>
              <li>• AR Visualization</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-4">Features</h4>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>• Character Recognition</li>
              <li>• Google Lens Style AR</li>
              <li>• Real-time Processing</li>
              <li>• Camera Capture</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-800 pt-6 text-center text-sm text-gray-400">
          <p>© 2024 Lipika Project. Built for Ranjana script preservation and digitization.</p>
        </div>
      </div>
    </footer>
  )
}

export default Footer

