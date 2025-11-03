"""
Legacy Flask app for backward compatibility
Prefer using ocr_service.py for production
"""

from flask import Flask, render_template

# Create directories for user datasets and uploads
import os
USER_DATA_DIR = os.path.join(os.getcwd(), "user_datasets")
USER_IMAGES_DIR = os.path.join(USER_DATA_DIR, "images")
os.makedirs(USER_IMAGES_DIR, exist_ok=True)

app = Flask(__name__)

# Import and initialize OCR service
import ocr_service
ocr_service.load_model()

@app.route('/')
def index():
    """Serve web interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint - delegates to OCR service"""
    from flask import request, jsonify
    return ocr_service.predict()

@app.route('/health')
def health():
    """Health check endpoint"""
    return ocr_service.health()

@app.route('/add_sample', methods=['POST'])
def add_sample():
    """Add labeled sample for training"""
    from flask import request, jsonify
    try:
        if 'image' not in request.files or 'label' not in request.form:
            return jsonify({'error': 'Image and label are required'}), 400
        
        file = request.files['image']
        label = request.form['label']
        
        if file.filename == '' or not label:
            return jsonify({'error': 'Invalid image or label'}), 400
        
        safe_name = os.path.basename(file.filename)
        save_path = os.path.join(USER_IMAGES_DIR, safe_name)
        file.save(save_path)
        
        labels_path = os.path.join(USER_DATA_DIR, 'labels.txt')
        with open(labels_path, 'a', encoding='utf-8') as f:
            f.write(f"{safe_name}|{ocr_service.normalize_unicode(label)}\n")
        
        return jsonify({'status': 'saved', 'image': safe_name, 'label': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_user', methods=['POST'])
def train_user():
    """Trigger training on user samples"""
    from flask import jsonify
    try:
        images = USER_IMAGES_DIR
        labels_path = os.path.join(USER_DATA_DIR, 'labels.txt')
        
        if not os.path.exists(labels_path):
            return jsonify({'error': 'No labels.txt found in user_datasets'}), 400
        
        import subprocess
        cmd = [
            'python', 'train_crnn_enhanced.py',
            '--images', images,
            '--labels', labels_path,
            '--epochs', '50'
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        
        if proc.returncode != 0:
            return jsonify({'error': 'Training failed', 'stderr': proc.stderr}), 500
        
        return jsonify({'status': 'training_completed', 'stdout': proc.stdout[-5000:]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting CRNN OCR Web App")
    print("=" * 40)
    
    if ocr_service.model is not None:
        print("Model loaded successfully!")
        print(f"Using device: {ocr_service.device}")
        print("Starting web server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train a model first.")
        print("Run: python cli.py train --data dataset")
