"""
Diagnostic script to understand what the model was trained on
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = "best_character_crnn_improved.pth"
checkpoint = torch.load(model_path, map_location=device)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_service_ar import ImprovedCharacterCRNN, CharacterCRNN

model_type = checkpoint.get('model_type', 'CharacterCRNN')
chars = checkpoint['chars']
num_classes = len(chars)

print("=" * 60)
print("MODEL DIAGNOSTIC")
print("=" * 60)
print(f"Model type: {model_type}")
print(f"Number of classes: {num_classes}")
print(f"\nCharacter set (first 30):")
for i, c in enumerate(chars[:30]):
    is_ascii = c.isascii() if c else False
    print(f"  [{i:2d}] = '{c}' (ASCII: {is_ascii})")

# Check if model predicts English or Ranjana
english_chars = [c for c in chars if c and c.isascii() and (c.isalpha() or c.isdigit())]
ranjana_chars = [c for c in chars if c and ord(c) > 127]

print(f"\nEnglish-like characters in model: {len(english_chars)}")
print(f"Ranjana characters in model: {len(ranjana_chars)}")

# Test on a real image
test_image = "../prepared_dataset/images/char_000001.png"
if os.path.exists(test_image):
    print(f"\nTesting on: {test_image}")
    img = Image.open(test_image).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    if model_type == 'ImprovedCharacterCRNN':
        model = ImprovedCharacterCRNN(num_classes=num_classes, img_height=64, img_width=64, dropout=0.5).to(device)
    else:
        model = CharacterCRNN(num_classes=num_classes).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        char_idx = predicted.item()
        conf = confidence.item()
        predicted_char = chars[char_idx] if char_idx < len(chars) else "?"
        
        print(f"Predicted: '{predicted_char}' (index {char_idx}, confidence {conf:.3f})")
        print(f"Expected label from dataset: 'cha'")
        print(f"Expected Ranjana: 'छ'")
        
        # Check top 5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        print(f"\nTop 5 predictions:")
        for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
            char = chars[idx.item()] if idx.item() < len(chars) else "?"
            print(f"  {i+1}. '{char}' (prob: {prob.item():.3f})")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
if len(ranjana_chars) > len(english_chars):
    print("Model appears to predict RANJANA characters")
    print("But dataset has ENGLISH transliteration labels")
    print("SOLUTION: Model needs to be retrained OR labels need conversion")
else:
    print("Model appears to predict ENGLISH transliteration")
    print("This matches the dataset labels")
    print("But character set contains Ranjana - CHECK THIS!")
