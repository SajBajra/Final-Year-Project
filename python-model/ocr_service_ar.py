"""
Lipika - AR-Ready OCR Service for Ranjana Script
Character-based recognition for AR visualization
Returns bounding boxes for each recognized character
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import unicodedata
import cv2
import numpy as np

# Optional: EasyOCR for improved character detection
# Note: EasyOCR may not work with Python 3.13+ due to scipy compatibility issues
# The service will work fine without it, using standard OpenCV segmentation
EASYOCR_AVAILABLE = False
if sys.version_info < (3, 13):
    try:
        import easyocr
        EASYOCR_AVAILABLE = True
        print("[INFO] EasyOCR available for enhanced character detection")
    except (ImportError, ModuleNotFoundError, Exception) as e:
        EASYOCR_AVAILABLE = False
        print(f"[INFO] EasyOCR not available: {type(e).__name__}")
        print("[INFO] Using standard OpenCV segmentation instead")
else:
    print("[INFO] Python 3.13+ detected - skipping EasyOCR due to scipy compatibility issues")
    print("[INFO] Using standard OpenCV segmentation (fully functional)")

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
    
    IMPORTANT: For single character images (like from training dataset),
    this function will return the entire image as a single segment.
    """
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image.convert('L'))
        original_height, original_width = img_array.shape
        
        # Only treat as single character if image is very small (training dataset size)
        # For larger images, always attempt segmentation
        is_single_char = (original_width < 80 and original_height < 80) and \
                        (abs(original_width - original_height) < 20)
        
        if is_single_char:
            print(f"[INFO] Detected small single character image ({original_width}x{original_height}), skipping segmentation")
            return [({'x': 0, 'y': 0, 'width': original_width, 'height': original_height}, image)]
        
        # For larger images, always attempt segmentation
        print(f"[INFO] Processing multi-character image ({original_width}x{original_height}), attempting segmentation")
        
        # Skip if image is too small
        if original_width < 20 or original_height < 20:
            print(f"[WARN] Image too small: {original_width}x{original_height}")
            # Try to process entire image as single character
            return [({'x': 0, 'y': 0, 'width': original_width, 'height': original_height}, image)]
        
        # IMPROVED preprocessing
        # 1. Apply adaptive thresholding for better handling of varying lighting
        binary = cv2.adaptiveThreshold(
            img_array, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # 2. Morphological operations to clean up noise (gentler)
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        
        # Close small gaps
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        # Remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # 3. Find contours with better parameters
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Better filtering parameters for multi-character images
        # Reduce minimum area threshold to catch more characters
        min_area = max(30, (original_width * original_height) * 0.0003)  # 0.03% of image area
        min_width = max(5, original_width * 0.01)  # At least 1% of width
        min_height = max(8, original_height * 0.03)  # At least 3% of height
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # More lenient filtering: check area, aspect ratio, and size
            if (area >= min_area and 
                w >= min_width and h >= min_height and
                h / w < 20 and w / h < 20):  # More lenient aspect ratio
                boxes.append((x, y, w, h))
        
        # If no boxes found, try more aggressive segmentation
        if not boxes:
            print("[WARN] No characters segmented with initial parameters, trying more aggressive segmentation")
            
            # Try with even more lenient parameters
            min_area = max(20, (original_width * original_height) * 0.0001)  # Very low threshold
            min_width = max(3, original_width * 0.005)
            min_height = max(5, original_height * 0.01)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if (area >= min_area and w >= min_width and h >= min_height and
                    h / w < 30 and w / h < 30):
                    boxes.append((x, y, w, h))
            
            if not boxes:
                print("[WARN] Still no characters found, processing entire image as single character")
                return [({'x': 0, 'y': 0, 'width': original_width, 'height': original_height}, image)]
            else:
                print(f"[INFO] Found {len(boxes)} characters with aggressive parameters")
        
        # Sort by x-coordinate (left to right), then by y (top to bottom for same column)
        boxes.sort(key=lambda b: (b[0], b[1]))
        
        # Extract crops with padding for better recognition
        results = []
        for x, y, w, h in boxes:
            # Add padding around character (15% on each side for better recognition)
            padding = max(5, int(min(w, h) * 0.15))
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(original_width - x_pad, w + 2 * padding)
            h_pad = min(original_height - y_pad, h + 2 * padding)
            
            try:
                crop = image.crop((x_pad, y_pad, x_pad + w_pad, y_pad + h_pad))
                bbox = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                results.append((bbox, crop))
            except Exception as e:
                print(f"[WARN] Error cropping character at ({x}, {y}): {e}")
                continue
        
        print(f"[INFO] Segmented {len(results)} characters from image")
        return results if results else [({'x': 0, 'y': 0, 'width': original_width, 'height': original_height}, image)]
        
    except Exception as e:
        print(f"[ERROR] Error in character segmentation: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return entire image as single character
        try:
            return [({'x': 0, 'y': 0, 'width': image.width, 'height': image.height}, image)]
        except:
            return []

# -------------------
# Utility Functions
# -------------------
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

def load_model():
    """Load the trained character model (supports both original and improved models)"""
    global model, chars
    
    try:
        # Load the IMPROVED model first (has Devanagari), fallback to old if not found
        # IMPORTANT: Load improved model first to get Devanagari characters!
        model_paths = ["best_character_crnn_improved.pth", "character_crnn_improved_final.pth", "best_character_crnn.pth"]
        checkpoint = None
        model_path = None
        
        # Try to find and load model file (prefer improved models)
        for path in model_paths:
            if os.path.exists(path):
                print(f"[INFO] Attempting to load model from: {path}")
                checkpoint = torch.load(path, map_location=device)
                model_path = path
                
                # Check if this model has Devanagari characters
                temp_chars = checkpoint.get('chars', [])
                ascii_count = len([c for c in temp_chars if c and c.isascii() and (c.isalpha() or c.isdigit())])
                unicode_count = len([c for c in temp_chars if c and ord(c) > 127])
                
                if unicode_count > 0 and ascii_count == 0:
                    print(f"[OK] Model has Devanagari characters ({unicode_count} Unicode, {ascii_count} ASCII) - using this model")
                    break
                elif ascii_count > 0:
                    print(f"[WARN] Model has ASCII characters ({ascii_count} ASCII, {unicode_count} Unicode) - this is an old model")
                    print(f"[WARN] Continuing to search for better model...")
                    # Don't break - continue searching for a better model
                    checkpoint = None
                    model_path = None
                else:
                    print(f"[INFO] Model has {len(temp_chars)} characters - using this model")
                    break
        
        if checkpoint is None:
            print("[ERROR] No character model found")
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
        print(f"[OK] Model loaded successfully! Type: {model_type}, Characters: {len(chars)}")
        
        # Print character set info for debugging
        ascii_chars = [c for c in chars if c and c.isascii() and (c.isalpha() or c.isdigit())]
        unicode_chars = [c for c in chars if c and ord(c) > 127]
        print(f"[INFO] Character set: {len(ascii_chars)} ASCII, {len(unicode_chars)} Unicode (Ranjana)")
        if len(ascii_chars) > 0:
            print(f"[WARN] Model contains ASCII characters: {ascii_chars[:10]}... (first 10)")
            print(f"[WARN] This may cause predictions to default to ASCII characters like 'a'")
        if len(unicode_chars) > 0:
            print(f"[INFO] Unicode characters available: {unicode_chars[:20]}... (first 20)")
        else:
            print(f"[ERROR] No Unicode (Ranjana) characters found in model! Model may not be trained for Ranjana script.")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
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

@app.route('/favicon.ico')
def favicon():
    """Suppress favicon 404 errors"""
    from flask import Response
    return Response(status=204)  # No Content

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
            print("[ERROR] Model not loaded")
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please ensure the model file exists.'
            }), 500
        
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image selected'
            }), 400
        
        print(f"[INFO] Processing image: {file.filename}")
        
        # Load image - read bytes first to avoid stream issues
        try:
            # Read file into bytes to avoid stream position issues
            file_bytes = file.read()
            if not file_bytes:
                return jsonify({
                    'success': False,
                    'error': 'Image file is empty'
                }), 400
            
            from io import BytesIO
            image = Image.open(BytesIO(file_bytes)).convert('L')
            print(f"[INFO] Image loaded: {image.size[0]}x{image.size[1]}, mode: {image.mode}")
        except Exception as e:
            print(f"[ERROR] Error loading image: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Error loading image: {str(e)}'
            }), 400
        
        # Segment into characters
        segments = segment_characters(image)
        
        if not segments:
            print("[WARN] No segments found, returning empty result")
            return jsonify({
                'success': True,
                'text': '',
                'characters': [],
                'message': 'No characters detected in image'
            })
        
        print(f"[INFO] Processing {len(segments)} character segments")
        
        # IMPORTANT: Use same normalization as training!
        # Training uses: mean=[0.485], std=[0.229]
        # This is critical for correct predictions!
        transform = transforms.Compose([
            transforms.Resize((64, 64), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Match training normalization
        ])
        
        # Predict each character
        results = []
        for i, (bbox, crop) in enumerate(segments):
            try:
                # Preprocess
                image_tensor = transform(crop).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(image_tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    
                    char_idx = predicted.item()
                    conf = confidence.item()
                
                # Get top 3 predictions for debugging
                top3_probs, top3_indices = torch.topk(probs, min(3, len(chars)), dim=1)
                top3_chars = [chars[idx.item()] if idx.item() < len(chars) else '?' for idx in top3_indices[0]]
                top3_conf = [prob.item() for prob in top3_probs[0]]
                
                # Accept predictions with reasonable confidence
                # Model should be trained with Devanagari labels, so predictions should be Devanagari
                if char_idx < len(chars):
                    char = chars[char_idx]
                    is_ascii_english = char and len(char) == 1 and char.isascii() and (char.isalpha() or char.isdigit())
                    
                    # If model predicts ASCII, something is wrong (model should only have Devanagari)
                    # But try to map it to Devanagari as fallback
                    if is_ascii_english:
                        print(f"[WARN] Character {i}: Model predicted ASCII '{char}' but model should only have Devanagari!")
                        print(f"[WARN] This suggests the model checkpoint may be misaligned or from old training")
                        print(f"[DEBUG] Top 3 predictions: {[(c, round(conf_val, 3)) for c, conf_val in zip(top3_chars, top3_conf)]}")
                        print(f"[DEBUG] Model has {len(chars)} characters, predicted index: {char_idx}")
                        
                        # Try to find Devanagari alternative in top predictions
                        found_devanagari = False
                        for alt_idx, alt_char in enumerate(top3_chars[1:], 1):  # Skip first (ASCII)
                            alt_conf = top3_conf[alt_idx]
                            # Check if this is a Devanagari character (Unicode > 127)
                            if alt_char and ord(alt_char) > 127:
                                # Found a Devanagari character in top predictions
                                if alt_conf > 0.1:  # Very low threshold to catch any Devanagari alternative
                                    char = alt_char
                                    conf = alt_conf
                                    found_devanagari = True
                                    print(f"[INFO] Using Devanagari alternative: '{char}' (confidence: {conf:.3f})")
                                    break
                        
                        # If no Devanagari alternative found, try transliteration mapping as last resort
                        if not found_devanagari:
                            try:
                                from transliteration_to_ranjana import TRANSLITERATION_TO_RANJANA
                                if char in TRANSLITERATION_TO_RANJANA:
                                    devanagari_char = TRANSLITERATION_TO_RANJANA[char]
                                    char = devanagari_char
                                    found_devanagari = True
                                    print(f"[INFO] Mapped ASCII '{chars[char_idx]}' to Devanagari '{char}' (fallback mapping)")
                                else:
                                    # Try lowercase version
                                    char_lower = char.lower()
                                    if char_lower in TRANSLITERATION_TO_RANJANA:
                                        devanagari_char = TRANSLITERATION_TO_RANJANA[char_lower]
                                        char = devanagari_char
                                        found_devanagari = True
                                        print(f"[INFO] Mapped ASCII '{chars[char_idx]}' (lowercase) to Devanagari '{char}'")
                                    else:
                                        print(f"[ERROR] No mapping found for ASCII character '{char}' - model prediction may be wrong")
                            except Exception as e:
                                print(f"[DEBUG] Could not load transliteration mapping: {e}")
                        
                        # If still no Devanagari found after mapping, warn and skip
                        if not found_devanagari:
                            print(f"[ERROR] Character {i}: Cannot map ASCII '{char}' to Devanagari - skipping (confidence: {conf:.3f})")
                            continue
                    elif conf < 0.2:
                        # If not ASCII but confidence is too low, try to find better alternative
                        print(f"[WARN] Character {i}: Low confidence ({conf:.3f}) for '{char}', checking alternatives")
                        found_better = False
                        for alt_idx, alt_char in enumerate(top3_chars[1:], 1):
                            alt_conf = top3_conf[alt_idx]
                            if alt_char and alt_conf > conf and alt_conf > 0.2:
                                # Found a better alternative
                                char = alt_char
                                conf = alt_conf
                                found_better = True
                                print(f"[INFO] Using better alternative: '{char}' (confidence: {conf:.3f})")
                                break
                        if not found_better:
                            print(f"[WARN] Character {i}: Skipping due to low confidence ({conf:.3f})")
                            continue
                    
                    # Add character if confidence is acceptable
                    # For dataset images (training data), model should have high confidence (>0.9)
                    # For real-world images, lower confidence is acceptable (>=0.15)
                    # We use a lower threshold to ensure dataset images are always recognized
                    min_confidence = 0.15  # Lower threshold to ensure dataset images are recognized
                    if conf >= min_confidence:
                        results.append({
                            'character': char,
                            'confidence': round(conf, 3),
                            'bbox': bbox,
                            'index': i
                        })
                        print(f"[INFO] Character {i}: '{char}' (confidence: {conf:.3f}, idx: {char_idx})")
                        # Warn if confidence is low (might be real-world image or misclassification)
                        if conf < 0.5:
                            print(f"[WARN] Low confidence prediction - might be real-world image or misclassification")
                            print(f"[DEBUG] Top 3: {[(c, round(conf_val, 3)) for c, conf_val in zip(top3_chars, top3_conf)]}")
                    else:
                        print(f"[WARN] Character {i}: Final confidence too low ({conf:.3f} < {min_confidence}), skipping")
                        print(f"[DEBUG] Top 3: {[(c, round(conf_val, 3)) for c, conf_val in zip(top3_chars, top3_conf)]}")
                else:
                    print(f"[WARN] Character {i}: Invalid index ({char_idx}/{len(chars)})")
                    print(f"[DEBUG] Top 3 predictions: {[(c, round(conf_val, 3)) for c, conf_val in zip(top3_chars, top3_conf)]}")
                    
            except Exception as e:
                print(f"[ERROR] Error predicting character {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Construct full text (Devanagari characters)
        text = ''.join([r['character'] for r in results])
        
        # Group characters into words based on spacing
        # In Devanagari, words are typically separated by spaces or punctuation
        words = []
        current_word = []
        word_bboxes = []
        current_word_bbox = None
        
        for i, result in enumerate(results):
            char = result['character']
            bbox = result.get('bbox', {})
            
            # Check if this character starts a new word
            # (space, punctuation, or significant horizontal gap indicates word boundary)
            is_word_boundary = False
            if char.strip() == '' or char in ['।', ',', '.', ';', ':', '!', '?']:
                is_word_boundary = True
            elif i > 0 and 'bbox' in results[i-1]:
                # Check for significant horizontal gap (word spacing)
                prev_bbox = results[i-1].get('bbox', {})
                if 'x' in bbox and 'x' in prev_bbox:
                    gap = bbox['x'] - (prev_bbox.get('x', 0) + prev_bbox.get('width', 0))
                    # If gap is more than 1.5x average character width, it's likely a word boundary
                    avg_width = sum([r.get('bbox', {}).get('width', 0) for r in results[:i+1]]) / max(i+1, 1)
                    if gap > avg_width * 1.5:
                        is_word_boundary = True
            
            if is_word_boundary and current_word:
                # Save current word
                word_text = ''.join(current_word)
                words.append({
                    'word': word_text,
                    'characters': current_word,
                    'bbox': current_word_bbox,
                    'character_count': len(current_word)
                })
                current_word = []
                current_word_bbox = None
            
            if char.strip() != '' and char not in ['।', ',', '.', ';', ':', '!', '?']:
                current_word.append(char)
                # Update word bounding box to encompass all characters
                if current_word_bbox is None:
                    current_word_bbox = bbox.copy() if bbox else {}
                else:
                    # Expand bbox to include this character
                    if 'x' in bbox:
                        current_word_bbox['x'] = min(current_word_bbox.get('x', bbox['x']), bbox['x'])
                    if 'y' in bbox:
                        current_word_bbox['y'] = min(current_word_bbox.get('y', bbox['y']), bbox['y'])
                    if 'width' in bbox and 'x' in bbox:
                        right_edge = max(
                            current_word_bbox.get('x', 0) + current_word_bbox.get('width', 0),
                            bbox['x'] + bbox.get('width', 0)
                        )
                        current_word_bbox['width'] = right_edge - current_word_bbox.get('x', 0)
                    if 'height' in bbox:
                        bottom_edge = max(
                            current_word_bbox.get('y', 0) + current_word_bbox.get('height', 0),
                            bbox['y'] + bbox.get('height', 0)
                        )
                        current_word_bbox['height'] = bottom_edge - current_word_bbox.get('y', 0)
        
        # Add final word if exists
        if current_word:
            word_text = ''.join(current_word)
            words.append({
                'word': word_text,
                'characters': current_word,
                'bbox': current_word_bbox,
                'character_count': len(current_word)
            })
        
        # If no words found but we have characters, treat all as one word
        if not words and results:
            words.append({
                'word': text,
                'characters': [r['character'] for r in results],
                'bbox': results[0].get('bbox', {}) if results else {},
                'character_count': len(results)
            })
        
        print(f"[INFO] Recognition complete: {len(results)} characters, {len(words)} words recognized")
        print(f"[INFO] Full text (Devanagari): '{text}'")
        if words:
            print(f"[INFO] Words: {[w['word'] for w in words]}")
        
        # If no results but we had segments, provide diagnostic info
        if not results and segments:
            print(f"[WARN] No characters recognized despite {len(segments)} segments found")
            print(f"[INFO] This may indicate: model confidence too low, image quality issues, or model mismatch")
        
        return jsonify({
            'success': True,
            'text': text,  # Full Devanagari text
            'characters': results,  # Individual character results
            'words': words,  # Word-level recognition
            'count': len(results),
            'word_count': len(words),
            'message': f'Recognized {len(results)} characters in {len(words)} words' if results else 'No characters detected with sufficient confidence'
        })
        
    except Exception as e:
        print(f"[ERROR] Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
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
        
        # IMPORTANT: Use same normalization as training!
        transform = transforms.Compose([
            transforms.Resize((64, 64), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Match training normalization
        ])
        
        for bbox, crop in segments:
            
            image_tensor = transform(crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probs = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                
                char_idx = predicted.item()
                conf = confidence.item()
                
                # Higher confidence threshold (0.5) to avoid false positives
                if char_idx < len(chars) and conf > 0.5:
                    char = chars[char_idx]
                    # CRITICAL: Filter out ALL ASCII English characters for Ranjana OCR
                    is_ascii_english = char and len(char) == 1 and char.isascii() and (char.isalpha() or char.isdigit())
                    if is_ascii_english:
                        continue  # Skip ALL ASCII predictions - we only want Ranjana characters
                    
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

@app.route('/count-characters', methods=['POST'])
def count_characters():
    """
    Dedicated endpoint for character count detection
    Uses improved segmentation + optional EasyOCR for accurate counting
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load image
        try:
            file_bytes = file.read()
            if not file_bytes:
                return jsonify({'error': 'Image file is empty'}), 400
            
            from io import BytesIO
            image = Image.open(BytesIO(file_bytes)).convert('L')
            print(f"[INFO] Character count request for image: {image.size[0]}x{image.size[1]}")
        except Exception as e:
            return jsonify({'error': f'Error loading image: {str(e)}'}), 400
        
        # Use EasyOCR if available for better detection
        if EASYOCR_AVAILABLE:
            try:
                reader = easyocr.Reader(['en', 'hi'], gpu=torch.cuda.is_available())
                img_array = np.array(image)
                results = reader.readtext(img_array)
                
                # Count detected text regions
                detected_regions = len(results)
                total_chars = sum(len(text) for text, _, _ in results)
                
                print(f"[INFO] EasyOCR detected {detected_regions} regions, {total_chars} characters")
                
                return jsonify({
                    'success': True,
                    'character_count': total_chars,
                    'text_regions': detected_regions,
                    'method': 'easyocr',
                    'detected_texts': [text for text, _, _ in results]
                })
            except Exception as e:
                print(f"[WARN] EasyOCR failed: {e}, falling back to standard segmentation")
        
        # Fallback to standard segmentation
        segments = segment_characters(image)
        character_count = len(segments)
        
        print(f"[INFO] Standard segmentation detected {character_count} characters")
        
        return jsonify({
            'success': True,
            'character_count': character_count,
            'text_regions': character_count,
            'method': 'standard_segmentation',
            'message': 'Character count estimated using segmentation'
        })
        
    except Exception as e:
        print(f"[ERROR] Error in character count endpoint: {e}")
        import traceback
        traceback.print_exc()
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

