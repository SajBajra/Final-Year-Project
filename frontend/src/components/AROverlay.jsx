import { useState, useEffect } from 'react'

const AROverlay = ({ image, characters }) => {
  const [imageUrl, setImageUrl] = useState(null)

  useEffect(() => {
    if (image instanceof File) {
      const reader = new FileReader()
      reader.onload = (e) => setImageUrl(e.target.result)
      reader.readAsDataURL(image)
    } else {
      setImageUrl(image)
    }
  }, [image])

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4 flex items-center">
        <span className="mr-3">👓</span>
        AR Visualization
      </h2>
      
      {imageUrl && (
        <div className="relative inline-block w-full">
          <img
            src={imageUrl}
            alt="AR Overlay"
            className="w-full h-auto rounded-xl shadow-lg"
          />
          
          {characters.map((char, idx) => {
            if (!char.bbox) return null
            
            return (
              <div
                key={idx}
                className="absolute border-2 border-blue-500 bg-blue-500 bg-opacity-20 hover:bg-opacity-30 transition-all cursor-pointer group"
                style={{
                  left: `${char.bbox.x || 0}px`,
                  top: `${char.bbox.y || 0}px`,
                  width: `${char.bbox.width || 0}px`,
                  height: `${char.bbox.height || 0}px`,
                }}
                title={`${char.character} (${char.confidence?.toFixed(1) || 'N/A'}%)`}
              >
                <div className="absolute -top-8 left-0 bg-blue-600 text-white px-2 py-1 rounded text-sm font-semibold opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
                  {char.character}
                </div>
              </div>
            )
          })}
        </div>
      )}
      
      <p className="text-sm text-gray-600 mt-4 text-center">
        Hover over highlighted boxes to see recognized characters
      </p>
    </div>
  )
}

export default AROverlay

