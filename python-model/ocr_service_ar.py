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
# Character CRNN Model
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
# Character Segmentation
# -------------------
def segment_characters(image):
    """
    Segment image into individual character bounding boxes
    Returns list of (bbox, crop) tuples
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image.convert('L'))
    
    # Threshold to binary
    _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter small noise
        if w > 5 and h > 10:
            boxes.append((x, y, w, h))
    
    # Sort by x-coordinate (left to right)
    boxes.sort(key=lambda b: b[0])
    
    # Extract crops
    results = []
    for x, y, w, h in boxes:
        crop = image.crop((x, y, x+w, y+h))
        bbox = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        results.append((bbox, crop))
    
    return results

# -------------------
# Utility Functions
# -------------------
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

def load_model():
    """Load the trained character model"""
    global model, chars
    
    try:
        if os.path.exists("best_character_crnn.pth"):
            checkpoint = torch.load("best_character_crnn.pth", map_location=device)
            chars = checkpoint['chars']
            num_classes = len(chars)
            model = CharacterCRNN(num_classes=num_classes).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"✓ Character model loaded with {len(chars)} characters")
            return True
        else:
            print("✗ No character model found")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

# -------------------
# API Endpoints
# -------------------
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

