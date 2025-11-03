# Lipika Frontend - Ranjana OCR System

Modern React web application with Google Lens-style AR overlay for Ranjana script recognition.

## Features

- ✨ **Modern UI** - Beautiful interface with Tailwind CSS
- 📸 **Image Upload** - Drag & drop or click to upload
- 📷 **Camera Capture** - Real-time webcam support
- 🔍 **OCR Recognition** - Character-level detection
- 👓 **AR Overlay** - Google Lens-style bounding boxes
- 📱 **Responsive** - Works on all devices

## Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Build

```bash
npm run build
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx          # App header
│   │   ├── Footer.jsx          # App footer
│   │   ├── ImageUpload.jsx     # File upload component
│   │   ├── CameraCapture.jsx   # Webcam capture
│   │   ├── OCRResult.jsx       # Results display
│   │   └── AROverlay.jsx       # AR visualization
│   ├── services/
│   │   └── ocrService.js       # API integration
│   ├── App.jsx                 # Main app
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── public/                     # Static assets
├── index.html                  # HTML template
├── package.json
├── vite.config.js             # Vite configuration
├── tailwind.config.js         # Tailwind CSS config
└── README.md
```

## API Integration

The frontend connects to the Python OCR service at `http://localhost:5000`:

- `POST /predict` - Upload image for OCR
- `GET /health` - Service health check

## Technologies

- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Axios** - HTTP client
- **React Webcam** - Camera integration

## Development

### Run Tests

```bash
npm run test
```

### Lint

```bash
npm run lint
```

## Deployment

### Build for Production

```bash
npm run build
```

Output will be in `dist/` directory.

### Deploy to GitHub Pages

```bash
npm run build
# Upload dist/ to GitHub Pages
```

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

MIT License
