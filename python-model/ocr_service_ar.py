"""
Lipika - AR-Ready OCR Service for Ranjana Script
Character-based recognition for Google Lens style AR visualization
Returns bounding boxes for each recognized character
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import unicodedata
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Global model variables
model = None
chars = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Character CRNN Model (Original)
# -------------------
class CharacterCRNN(torch.nn.Module):
    """Character-based CRNN for individual character recognition"""
    
    def __init__(self, num_classes, img_height=64, img_width=64):
        super(CharacterCRNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.rnn = torch.nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        conv_features = self.cnn(x)
        b, c, h, w = conv_features.size()
        conv_features = conv_features.view(b, c, -1).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv_features)
        output = self.classifier(rnn_out.squeeze(1))
        return output

# -------------------
# Improved Character CRNN Model
# -------------------
class ImprovedCharacterCRNN(torch.nn.Module):
    """
    Improved Character-based CRNN with better architecture
    - Deeper CNN with residual connections
    - Attention mechanism
    - Better regularization
    """
    def __init__(self, num_classes, img_height=64, img_width=64, dropout=0.5):
        super(ImprovedCharacterCRNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # Improved CNN Feature Extractor with residual connections
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
        )
        self.conv1_res = torch.nn.Conv2d(1, 32, 1)  # Skip connection
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
        )
        
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
        )
        
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        # Global pooling
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.Sigmoid()
        )
        
        # Improved RNN
        self.rnn = torch.nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if dropout > 0 else 0
        )
        
        # Improved Classification head with more layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # First conv with residual
        out = self.conv1(x) + self.conv1_res(x)
        
        # Continue through CNN
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        
        # Global pooling
        conv_features = self.global_pool(out)  # [B, 256, 1, 1]
        conv_features = conv_features.squeeze(-1).squeeze(-1)  # [B, 256]
        
        # Apply attention
        attention_weights = self.attention(conv_features)  # [B, 256]
        conv_features = conv_features * attention_weights
        
        # Reshape for RNN: [B, C] -> [B, 1, C]
        conv_features = conv_features.unsqueeze(1)
        
        # RNN processing
        rnn_out, _ = self.rnn(conv_features)
        
        # Classification
        output = self.classifier(rnn_out.squeeze(1))
        
        return output

# -------------------
# Character Segmentation
# -------------------
def segment_characters(image):
    """
    IMPROVED character segmentation with better preprocessing
    Returns list of (bbox, crop) tuples
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image.convert('L'))
    original_height, original_width = img_array.shape
    
    # IMPROVED preprocessing
    # 1. Apply adaptive thresholding for better handling of varying lighting
    binary = cv2.adaptiveThreshold(
        img_array, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 2
    )
    
    # 2. Morphological operations to clean up noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 3. Find contours with better parameters
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Get bounding boxes with improved filtering
    boxes = []
    min_area = (original_width * original_height) * 0.001  # 0.1% of image area
    min_width = max(5, original_width * 0.01)  # At least 1% of width
    min_height = max(10, original_height * 0.05)  # At least 5% of height
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Better filtering: check area, aspect ratio, and size
        if (area >= min_area and 
            w >= min_width and h >= min_height and
            h / w < 10 and w / h < 10):  # Reasonable aspect ratio
            boxes.append((x, y, w, h))
    
    if not boxes:
        return []
    
    # Sort by x-coordinate (left to right), then by y (top to bottom for same column)
    boxes.sort(key=lambda b: (b[0], b[1]))
    
    # Extract crops with padding for better recognition
    results = []
    for x, y, w, h in boxes:
        # Add padding around character (10% on each side)
        padding = max(3, int(min(w, h) * 0.1))
        x_pad = max(0, x - padding)
        y_pad = max(0, y - padding)
        w_pad = min(original_width - x_pad, w + 2 * padding)
        h_pad = min(original_height - y_pad, h + 2 * padding)
        
        crop = image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
        bbox = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        results.append((bbox, crop))
    
    return results

# -------------------
# Utility Functions
# -------------------
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

def load_model():
    """Load the trained character model (supports both original and improved models)"""
    global model, chars
    
    try:
        model_paths = ["best_character_crnn.pth", "best_character_crnn_improved.pth"]
        checkpoint = None
        model_path = None
        
        # Try to find and load model file
        for path in model_paths:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=device)
                model_path = path
                break
        
        if checkpoint is None:
            print("✗ No character model found")
            return False
        
        # Get model type from checkpoint (default to original)
        model_type = checkpoint.get('model_type', 'CharacterCRNN')
        chars = checkpoint['chars']
        num_classes = len(chars)
        
        # Load appropriate model class
        if model_type == 'ImprovedCharacterCRNN':
            print(f"Loading ImprovedCharacterCRNN model from {model_path}...")
            model = ImprovedCharacterCRNN(num_classes=num_classes, img_height=64, img_width=64, dropout=0.5).to(device)
        else:
            print(f"Loading CharacterCRNN model from {model_path}...")
            model = CharacterCRNN(num_classes=num_classes).to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model loaded successfully! Type: {model_type}, Characters: {len(chars)}")
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# -------------------
# API Endpoints
# -------------------
@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API information"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lipika - AR OCR Service</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #4a90e2;
                border-bottom: 3px solid #4a90e2;
                padding-bottom: 10px;
            }
            .endpoint {
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #4a90e2;
                border-radius: 5px;
            }
            .method {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
                margin-right: 10px;
            }
            .get { background: #28a745; color: white; }
            .post { background: #007bff; color: white; }
            .status {
                padding: 10px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
            code {
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            a {
                color: #4a90e2;
                text-decoration: none;
            }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📜 Lipika - AR-Ready OCR Service</h1>
            <p><strong>Ranjana Script OCR with AR Bounding Boxes</strong></p>
            
            <div class="status ''' + ('success' if model is not None else 'error') + '''">
                <strong>Service Status:</strong> ''' + ('✅ Model Loaded' if model is not None else '❌ Model Not Loaded') + '''<br>
                <strong>Device:</strong> ''' + str(device) + '''<br>
                <strong>Characters:</strong> ''' + str(len(chars) if chars else 0) + '''
            </div>
            
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong><a href="/health">/health</a></strong>
                <p>Health check endpoint. Returns service status and model information.</p>
                <code>curl http://localhost:5000/health</code>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/predict</strong>
                <p>OCR prediction with AR bounding boxes. Accepts multipart/form-data with 'image' field.</p>
                <code>curl -X POST -F "image=@your_image.png" http://localhost:5000/predict</code>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/predict/base64</strong>
                <p>OCR prediction with base64-encoded image. Accepts JSON with 'image' field (base64 string).</p>
                <code>curl -X POST -H "Content-Type: application/json" -d '{"image":"base64_string"}' http://localhost:5000/predict/base64</code>
            </div>
            
            <h2>Response Format</h2>
            <pre style="background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto;">
{
  "success": true,
  "text": "Recognized text...",
  "characters": [
    {
      "character": "न",
      "confidence": 0.987,
      "bbox": {"x": 10, "y": 5, "width": 25, "height": 30},
      "index": 0
    }
  ],
  "count": 1
}</pre>
            
            <p style="margin-top: 30px; color: #666; font-size: 0.9em;">
                For the React frontend, go to <a href="http://localhost:5173">http://localhost:5173</a>
            </p>
        </div>
    </body>
    </html>
    '''

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'chars_count': len(chars) if chars else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    OCR prediction with AR bounding boxes
    Returns recognized text with character positions
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load image
        image = Image.open(file.stream).convert('L')
        
        # Segment into characters
        segments = segment_characters(image)
        
        if not segments:
            return jsonify({
                'success': True,
                'text': '',
                'characters': [],
                'message': 'No characters detected'
            })
        
        # Predict each character
        results = []
        for i, (bbox, crop) in enumerate(segments):
            # Preprocess
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            image_tensor = transform(crop).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(image_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                char_idx = predicted.item()
                conf = confidence.item()
                
                if char_idx < len(chars) and conf > 0.5:
                    char = chars[char_idx]
                    results.append({
                        'character': char,
                        'confidence': round(conf, 3),
                        'bbox': bbox,
                        'index': i
                    })
        
        # Construct full text
        text = ''.join([r['character'] for r in results])
        
        return jsonify({
            'success': True,
            'text': text,
            'characters': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """OCR endpoint for base64-encoded images"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        import base64
        import io
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        
        # Segment and predict
        segments = segment_characters(image)
        results = []
        
        for bbox, crop in segments:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            image_tensor = transform(crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                char_idx = predicted.item()
                conf = confidence.item()
                
                if char_idx < len(chars) and conf > 0.5:
                    char = chars[char_idx]
                    results.append({
                        'character': char,
                        'confidence': round(conf, 3),
                        'bbox': bbox
                    })
        
        text = ''.join([r['character'] for r in results])
        
        return jsonify({
            'success': True,
            'text': text,
            'characters': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("AR-Ready Ranjana Script OCR Service")
    print("=" * 60)
    
    if load_model():
        print(f"Device: {device}")
        print("Service running on http://0.0.0.0:5000")
        print("Endpoints:")
        print("  GET  /health         - Health check")
        print("  POST /predict        - OCR with bounding boxes")
        print("  POST /predict/base64 - OCR with bounding boxes (base64)")
        print("=" * 60)
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train a character model first.")
        print("Run: python train_character_crnn.py")

