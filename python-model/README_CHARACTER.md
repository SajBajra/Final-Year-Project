# Lipika - Character-Based AR OCR System

## Overview
**Lipika** (लिपिका) is an **AR-ready OCR service** for Ranjana script, optimized for Google Lens-style augmented reality visualization. Unlike word-based OCR, this system recognizes individual characters with bounding box information for AR overlay.

## Why Character-Based?

### For Google Lens Style AR
- ✅ **Bounding boxes** for each character → AR overlay
- ✅ **Character-level precision** → Better placement
- ✅ **Flexible recognition** → Handles any text length
- ✅ **Translation-ready** → Easy to replace individual chars

### Architecture
```
Image → Character Segmentation → Individual Recognition → AR Bounding Boxes
```

## Quick Start

### 1. Train Character Model
```bash
cd python-model
python train_character_crnn.py \
  --images ../char_dataset/images \
  --train_labels ../char_dataset/train_labels.txt \
  --val_labels ../char_dataset/val_labels.txt \
  --epochs 100 \
  --batch_size 64
```

### 2. Run AR OCR Service
```bash
python ocr_service_ar.py
```

### 3. Test API
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@test_image.png"
```

## API Response Format

### Standard Response
```json
{
  "success": true,
  "text": "नेपाली भाषा",
  "characters": [
    {
      "character": "न",
      "confidence": 0.987,
      "bbox": {"x": 10, "y": 5, "width": 25, "height": 30},
      "index": 0
    },
    {
      "character": "े",
      "confidence": 0.954,
      "bbox": {"x": 35, "y": 5, "width": 15, "height": 30},
      "index": 1
    }
    // ... more characters
  ],
  "count": 15
}
```

## Comparison: Character vs Word

### Word-Based (Old)
```json
{
  "text": "नेपाली भाषा",
  "detected": true
}
```
❌ No bounding boxes  
❌ Can't do AR overlay  
❌ Limited flexibility  

### Character-Based (New) ✅
```json
{
  "text": "नेपाली भाषा",
  "characters": [
    {"character": "न", "bbox": {...}},
    {"character": "े", "bbox": {...}},
    // ... with positions
  ]
}
```
✅ Bounding boxes for AR  
✅ Character-level control  
✅ Translation-ready  
✅ Google Lens style  

## Model Details

### Character CRNN Architecture
- **Input**: 64×64 grayscale character images
- **CNN**: 5-layer feature extractor (32→64→128→256→512)
- **RNN**: 2-layer bidirectional LSTM
- **Classifier**: 3-layer FC with dropout
- **Output**: 82 character classes

### Character Set
- **Total**: 82 characters
- **Vowels**: अ, आ, इ, ई, उ, ऊ, ए, ऐ, ओ, औ
- **Consonants**: All Ranjana consonants
- **Diacritics**: ँ, ं, ः, ा, ि, ी, ु, ू, े, ो, ्
- **Numbers**: ०-९
- **Punctuation**: ।, ॥

## AR Integration Example

### React/JavaScript
```javascript
const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
});

const data = await response.json();

// Overlay on original image
data.characters.forEach((char, i) => {
    const { x, y, width, height } = char.bbox;
    
    // Create AR overlay
    const overlay = document.createElement('div');
    overlay.style.position = 'absolute';
    overlay.style.left = `${x}px`;
    overlay.style.top = `${y}px`;
    overlay.style.width = `${width}px`;
    overlay.style.height = `${height}px`;
    overlay.style.border = '2px solid blue';
    overlay.textContent = char.character;
    
    // Add translation if needed
    // overlay.textContent = translate(char.character);
    
    imageContainer.appendChild(overlay);
});
```

### Python
```python
import requests

response = requests.post('http://localhost:5000/predict', 
                        files={'image': open('image.png', 'rb')})
data = response.json()

for char_data in data['characters']:
    bbox = char_data['bbox']
    char = char_data['character']
    
    # Overlay bounding box
    # cv2.rectangle(img, (bbox['x'], bbox['y']), 
    #               (bbox['x']+bbox['width'], bbox['y']+bbox['height']), 
    #               (0,255,0), 2)
```

## Files

- `train_character_crnn.py` - Training script for character model
- `ocr_service_ar.py` - AR-ready OCR service with bounding boxes
- `char_dataset/` - 164K character images with labels
- `best_character_crnn.pth` - Trained model (after training)

## Training Data

### Dataset Stats
- **Total images**: 164,000
- **Characters**: 82 classes
- **Training**: 131,200 images
- **Validation**: 32,800 images
- **Images per char**: 2,000

### Format
```
char_033_0619.png|अ
char_045_1552.png|ञ
char_081_1950.png|॥
```

## Next Steps for Google Lens

1. ✅ **Character recognition** - Done
2. ⏳ **Translation API** - Add translation service
3. ⏳ **AR overlay** - Frontend visualization
4. ⏳ **Camera integration** - Real-time capture
5. ⏳ **Text-to-speech** - Voice readout

## Performance

Expected metrics (after training):
- **Training accuracy**: 90-95%
- **Validation accuracy**: 85-90%
- **Inference speed**: ~50-100ms per image
- **Memory**: ~200MB for model + weights

---

**Status**: Ready for character-based training and AR integration

