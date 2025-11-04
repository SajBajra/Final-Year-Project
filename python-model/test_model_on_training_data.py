"""
Test script to verify model accuracy on training dataset
This helps identify preprocessing/normalization issues
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json
from transliteration_to_ranjana import transliterate_to_ranjana

# Load model and chars
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Try to load model
model_path = "best_character_crnn_improved.pth"
if not os.path.exists(model_path):
    print(f"[ERROR] Model not found: {model_path}")
    sys.exit(1)

print(f"[INFO] Loading model from {model_path}...")
checkpoint = torch.load(model_path, map_location=device)

# Import model class
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_service_ar import ImprovedCharacterCRNN, CharacterCRNN

model_type = checkpoint.get('model_type', 'CharacterCRNN')
chars = checkpoint['chars']
num_classes = len(chars)

if model_type == 'ImprovedCharacterCRNN':
    model = ImprovedCharacterCRNN(num_classes=num_classes, img_height=64, img_width=64, dropout=0.5).to(device)
else:
    model = CharacterCRNN(num_classes=num_classes).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"[OK] Model loaded: {model_type}, {num_classes} classes")
print(f"[INFO] Character set: {''.join(chars[:20])}... ({len(chars)} total)")

# Test transform (MUST match training)
transform = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Match training
])

# Test on a few images from the dataset
# Use Ranjana labels if available, otherwise use original
dataset_images = "../prepared_dataset/images"
if os.path.exists("../prepared_dataset/train_labels_ranjana.txt"):
    dataset_labels = "../prepared_dataset/train_labels_ranjana.txt"
    print("[INFO] Using Ranjana labels file")
else:
    dataset_labels = "../prepared_dataset/train_labels.txt"
    print("[INFO] Using original labels file (will convert during testing)")

if not os.path.exists(dataset_images):
    print(f"[ERROR] Dataset images not found: {dataset_images}")
    sys.exit(1)

if not os.path.exists(dataset_labels):
    print(f"[ERROR] Labels file not found: {dataset_labels}")
    sys.exit(1)

# Load labels (handle both tab and pipe separator)
with open(dataset_labels, 'r', encoding='utf-8') as f:
    labels_data = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Try tab separator first, then pipe
        if '\t' in line:
            parts = line.split('\t')
        elif '|' in line:
            parts = line.split('|')
        else:
            # Assume format: filename.ext label
            parts = line.rsplit(' ', 1)
        
        if len(parts) >= 2:
            labels_data.append((parts[0].strip(), parts[1].strip()))

print(f"\n[INFO] Testing on {len(labels_data)} images from training set...\n")

correct = 0
total = 0
errors = []

# Test first 50 images
test_count = min(50, len(labels_data))

for idx, (img_file, label) in enumerate(labels_data[:test_count]):
    img_path = os.path.join(dataset_images, img_file)
    
    if not os.path.exists(img_path):
        continue
    
    try:
        # Load and preprocess image exactly like inference
        img = Image.open(img_path).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            char_idx = predicted.item()
            conf = confidence.item()
        
        # Get predicted character
        if char_idx < len(chars):
            predicted_char = chars[char_idx]
            expected_label = label.strip()
            
            # Convert English transliteration to Ranjana script
            expected_char = transliterate_to_ranjana(expected_label)
            
            is_correct = (predicted_char == expected_char)
            
            if is_correct:
                correct += 1
                status = "[OK]"
            else:
                status = "[X]"
                errors.append({
                    'file': img_file,
                    'expected_label': expected_label,
                    'expected_char': expected_char,
                    'predicted': predicted_char,
                    'confidence': conf
                })
            
            total += 1
            
            # Print every 10th result or if incorrect
            if (idx + 1) % 10 == 0 or not is_correct:
                expected_display = f"'{expected_label}'->'{expected_char}'" if expected_label != expected_char else f"'{expected_char}'"
                print(f"{status} {idx+1}/{test_count}: Expected={expected_display} Predicted='{predicted_char}' Conf={conf:.3f}")
        else:
            print(f"[ERROR] Invalid char index: {char_idx}/{len(chars)}")
            
    except Exception as e:
        print(f"[ERROR] Error processing {img_file}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print(f"Results: {correct}/{total} correct ({100*correct/total:.2f}%)")
print(f"{'='*60}")

if errors:
    print(f"\n[ERROR] Found {len(errors)} errors:")
    for err in errors[:10]:  # Show first 10 errors
        expected_str = f"'{err['expected_label']}'->'{err['expected_char']}'" if err['expected_label'] != err['expected_char'] else f"'{err['expected_char']}'"
        print(f"  {err['file']}: Expected {expected_str} but got '{err['predicted']}' (conf: {err['confidence']:.3f})")
    
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")

# Save detailed results
results_file = "model_test_results.json"
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        'accuracy': 100 * correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'errors': errors
    }, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Detailed results saved to: {results_file}")

if correct / total < 0.8:
    print("\n[WARNING] Model accuracy on training data is < 80%!")
    print("This indicates a preprocessing or normalization mismatch.")
    print("Please check:")
    print("  1. Normalization values match training (mean=[0.485], std=[0.229])")
    print("  2. Image resize matches training (64x64)")
    print("  3. Model is loaded correctly")
    sys.exit(1)
else:
    print("\n[OK] Model performs well on training data!")
