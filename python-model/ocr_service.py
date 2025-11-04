"""
Ranjana Script OCR Service
A production-ready REST API service for Ranjana text recognition using CRNN model.
This service exposes endpoints for image OCR that can be called from external systems.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import unicodedata

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global model variables
model = None
chars = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Enhanced CRNN Model Architecture
# -------------------
class EnhancedCRNN(torch.nn.Module):
    """CRNN model for Ranjana script word/sequence recognition"""
    
    def __init__(self, num_classes, img_height=32, img_width=128):
        super(EnhancedCRNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN Feature Extractor
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 1), (2, 1)),
            
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 1), (2, 1)),
            
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, 2, 1, 0),
            torch.nn.ReLU(True),
        )
        
        # Bidirectional LSTM for sequence modeling
        self.rnn = torch.nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Character classification head
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"Height after CNN must be 1, got {h}"
        
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        
        recurrent, _ = self.rnn(conv)
        output = self.fc(recurrent)
        output = output.permute(1, 0, 2)
        
        return output

# -------------------
# Utility Functions
# -------------------
def ctc_decode(output, chars, blank_idx=0):
    """Decode CTC output to text string"""
    probs = F.softmax(output, dim=2)
    argmax = probs.argmax(2)
    
    batch_size = argmax.size(1)
    decoded_strings = []
    
    for b in range(batch_size):
        sequence = argmax[:, b].cpu().numpy()
        decoded = []
        prev = -1
        
        for idx in sequence:
            if idx != prev and idx != blank_idx:
                if idx < len(chars):
                    decoded.append(chars[idx])
            prev = idx
        
        decoded_strings.append(''.join(decoded))
    
    return decoded_strings

def ctc_beam_search(output, chars, blank_idx=0, beam_width=10):
    """Beam search decoding for better accuracy"""
    T, B, C = output.size()
    log_probs = F.log_softmax(output, dim=2)
    results = []
    
    for b in range(B):
        beams = {(): 0.0}
        for t in range(T):
            next_beams = {}
            lp = log_probs[t, b]
            topk = torch.topk(lp, k=min(beam_width, C))[1].tolist()
            
            for path, score in beams.items():
                for k in topk:
                    k = int(k)
                    p = score + float(lp[k])
                    if k == blank_idx:
                        next_beams[path] = max(next_beams.get(path, -1e9), p)
                    else:
                        if len(path) > 0 and path[-1] == k:
                            new_path = path
                        else:
                            new_path = path + (k,)
                        next_beams[new_path] = max(next_beams.get(new_path, -1e9), p)
            
            beams = dict(sorted(next_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width])
        
        best_path = max(beams.items(), key=lambda x: x[1])[0]
        s = []
        prev = -1
        for k in best_path:
            if k != prev and k != blank_idx and k < len(chars):
                s.append(chars[k])
            prev = k
        results.append(''.join(s))
    
    return results

def postprocess_text(text: str) -> str:
    """Post-process recognized text"""
    s = normalize_unicode(text)
    s = ' '.join(s.split())  # Collapse whitespace
    return s

def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form"""
    return unicodedata.normalize('NFC', text)

def load_model():
    """Load trained model from checkpoint"""
    global model, chars
    
    try:
        # Try loading enhanced model
        if os.path.exists("enhanced_crnn_model.pth"):
            checkpoint = torch.load("enhanced_crnn_model.pth", map_location=device)
            chars = checkpoint['chars']
            num_classes = len(chars)
            model = EnhancedCRNN(num_classes=num_classes).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"[OK] Enhanced model loaded with {len(chars)} characters")
            return True
        elif os.path.exists("best_crnn_model.pth"):
            checkpoint = torch.load("best_crnn_model.pth", map_location=device)
            chars = checkpoint['chars']
            num_classes = len(chars)
            model = EnhancedCRNN(num_classes=num_classes).to(device)
            state = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state)
            model.eval()
            print(f"[OK] Basic model loaded with {len(chars)} characters")
            return True
        else:
            print("[ERROR] No trained model found")
            return False
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
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
    Main OCR prediction endpoint
    Accepts: multipart/form-data with 'image' field
    Returns: JSON with recognized text
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load and preprocess image
        image = Image.open(file.stream).convert("L")
        
        transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            preds = ctc_beam_search(output, chars, blank_idx=0, beam_width=15)
            prediction = preds[0] if preds else ctc_decode(output, chars)[0]
            prediction = postprocess_text(prediction)
        
        return jsonify({
            'success': True,
            'text': prediction if prediction else '',
            'detected': len(prediction) > 0
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """
    OCR endpoint that accepts base64-encoded images
    Useful for mobile apps and JavaScript
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        import base64
        import io
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert("L")
        
        transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            preds = ctc_beam_search(output, chars, blank_idx=0, beam_width=15)
            prediction = preds[0] if preds else ctc_decode(output, chars)[0]
            prediction = postprocess_text(prediction)
        
        return jsonify({
            'success': True,
            'text': prediction if prediction else '',
            'detected': len(prediction) > 0
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Ranjana Script OCR Service")
    print("=" * 60)
    
    if load_model():
        print(f"Device: {device}")
        print("Service running on http://0.0.0.0:5000")
        print("Endpoints:")
        print("  GET  /health         - Health check")
        print("  POST /predict        - OCR prediction (multipart)")
        print("  POST /predict/base64 - OCR prediction (base64)")
        print("=" * 60)
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train a model first.")
        print("Run: python cli.py train --data dataset")

