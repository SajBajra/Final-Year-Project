# React Frontend - View Layer

## Overview
Modern React web application that provides the user interface for the Ranjana OCR system, similar to Google Lens.

## Features
- Image upload and camera capture
- Real-time OCR processing
- Display recognized text
- AR overlay visualization (future)
- Translation display (future)
- Modern, responsive UI

## Technology Stack
- React 18+
- TypeScript (recommended)
- Vite or Create React App
- Tailwind CSS or Material-UI
- React Webcam (for camera capture)

## Setup

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation
```bash
# Using Vite
npm create vite@latest .
npm install

# Or Create React App
npx create-react-app .
npm install

# Install dependencies
npm install axios react-webcam @mui/material @emotion/react @emotion/styled
```

### Run Development Server
```bash
npm run dev    # Vite
# or
npm start      # Create React App
```

## Project Structure (To Be Created)
```
frontend/
├── src/
│   ├── components/
│   │   ├── ImageUpload.jsx      # File upload component
│   │   ├── CameraCapture.jsx    # Webcam capture
│   │   ├── OCRResult.jsx        # Display results
│   │   └── ARVisualization.jsx  # AR overlay (future)
│   ├── services/
│   │   └── ocrService.js        # API calls to Java backend
│   ├── App.jsx                  # Main app component
│   ├── App.css                  # Global styles
│   └── main.jsx                 # Entry point
├── public/
├── package.json
├── vite.config.js              # Vite config
└── README.md
```

## API Integration

Example code to call Java backend:
```javascript
// services/ocrService.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8080/api';

export const recognizeText = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await axios.post(
    `${API_BASE_URL}/ocr`, 
    formData, 
    {
      headers: { 'Content-Type': 'multipart/form-data' }
    }
  );
  
  return response.data;
};
```

## Component Examples

### Image Upload
```javascript
function ImageUpload({ onImageSelect }) {
  return (
    <div>
      <input 
        type="file" 
        accept="image/*" 
        onChange={(e) => onImageSelect(e.target.files[0])}
      />
    </div>
  );
}
```

### OCR Result Display
```javascript
function OCRResult({ text, loading }) {
  if (loading) return <div>Processing...</div>;
  return (
    <div className="result">
      <h2>Recognized Text:</h2>
      <p>{text || 'No text detected'}</p>
    </div>
  );
}
```

## Future Enhancements
- [ ] AR overlay on captured images
- [ ] Translation integration
- [ ] Text-to-speech
- [ ] History of OCR results
- [ ] Export functionality
- [ ] Mobile responsiveness

---

**Status**: Ready for implementation

