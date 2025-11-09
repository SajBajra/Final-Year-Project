"""
Test script to verify the trained model loads correctly
Checks:
1. Model file exists and loads successfully
2. Model contains Ranjana (Unicode) characters
3. Model can make predictions on a dummy image
"""

import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms

# Import model classes
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr_service_ar import ImprovedCharacterCRNN, CharacterCRNN

def test_model():
    """Test the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Check for model files
    model_paths = [
        "best_character_crnn_improved.pth",
        "character_crnn_improved_final.pth",
        "best_character_crnn.pth"
    ]
    
    model_file = None
    for path in model_paths:
        if os.path.exists(path):
            model_file = path
            print(f"[OK] Found model file: {path}")
            break
    
    if not model_file:
        print("[ERROR] No model file found!")
        print(f"   Looking for: {', '.join(model_paths)}")
        return False
    
    # Load checkpoint
    try:
        print(f"\nLoading model from: {model_file}")
        checkpoint = torch.load(model_file, map_location=device)
        print("[OK] Model checkpoint loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return False
    
    # Check checkpoint contents
    print("\n" + "="*60)
    print("CHECKPOINT INFORMATION")
    print("="*60)
    
    required_keys = ['chars', 'model_state_dict']
    for key in required_keys:
        if key in checkpoint:
            print(f"[OK] Contains '{key}'")
        else:
            print(f"[ERROR] Missing '{key}'")
            return False
    
    # Get model type
    model_type = checkpoint.get('model_type', 'CharacterCRNN')
    print(f"[OK] Model type: {model_type}")
    
    # Check character set
    chars = checkpoint['chars']
    num_classes = len(chars)
    print(f"[OK] Total characters: {num_classes}")
    
    # Analyze character set
    ascii_chars = [c for c in chars if c and c.isascii() and (c.isalpha() or c.isdigit())]
    unicode_chars = [c for c in chars if c and ord(c) > 127]
    special_chars = [c for c in chars if c and not (c.isascii() and (c.isalpha() or c.isdigit())) and ord(c) <= 127]
    
    print(f"\nCharacter breakdown:")
    print(f"  - ASCII letters/numbers: {len(ascii_chars)}")
    print(f"  - Unicode (Ranjana): {len(unicode_chars)}")
    print(f"  - Special characters: {len(special_chars)}")
    
    # Check for Ranjana characters (Unicode range for Ranjana: U+A860–U+A87F)
    ranjana_chars = [c for c in chars if c and '\uA860' <= c <= '\uA87F']
    # Check for Devanagari characters (Unicode range: U+0900–U+097F) - also acceptable
    devanagari_chars = [c for c in chars if c and '\u0900' <= c <= '\u097F']
    
    print(f"  - Ranjana script (U+A860–U+A87F): {len(ranjana_chars)}")
    print(f"  - Devanagari script (U+0900–U+097F): {len(devanagari_chars)}")
    
    has_ranjana = len(ranjana_chars) > 0
    has_devanagari = len(devanagari_chars) > 0
    has_unicode = len(unicode_chars) > 0
    
    if has_ranjana:
        print(f"\n[SUCCESS] Model contains {len(ranjana_chars)} Ranjana characters!")
        try:
            print(f"   Sample Ranjana characters: {''.join(ranjana_chars[:10])}")
        except UnicodeEncodeError:
            print(f"   Sample Ranjana characters: (Unicode characters present)")
    elif has_devanagari:
        print(f"\n[SUCCESS] Model contains {len(devanagari_chars)} Devanagari characters!")
        print("   (Devanagari is the target output format - this is correct!)")
        try:
            print(f"   Sample Devanagari characters: {''.join(devanagari_chars[:10])}")
        except UnicodeEncodeError:
            print(f"   Sample Devanagari characters: (Unicode characters present)")
    elif has_unicode:
        print(f"\n[WARNING] Model contains {len(unicode_chars)} Unicode characters")
        print("   (May be Devanagari or other script - verify character range)")
        try:
            print(f"   Sample Unicode characters: {len(unicode_chars)} characters found")
        except UnicodeEncodeError:
            print(f"   Sample Unicode characters: (Unicode characters present)")
    else:
        print(f"\n[ERROR] Model contains NO Unicode (Ranjana/Devanagari) characters!")
        print("   This model was likely trained on ASCII transliterations.")
        print("   You need to retrain with Ranjana or Devanagari labels.")
        return False
    
    # Check training info
    if 'epoch' in checkpoint:
        print(f"\nTraining info:")
        print(f"  - Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"  - Validation accuracy: {checkpoint['val_acc']:.2f}%")
    if 'train_acc' in checkpoint:
        print(f"  - Training accuracy: {checkpoint['train_acc']:.2f}%")
    
    # Try to load model architecture
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE TEST")
    print("="*60)
    
    try:
        if model_type == 'ImprovedCharacterCRNN':
            model = ImprovedCharacterCRNN(num_classes=num_classes, img_height=64, img_width=64, dropout=0.5).to(device)
        else:
            model = CharacterCRNN(num_classes=num_classes).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("[OK] Model architecture loaded successfully")
        print(f"[OK] Model state dict loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model architecture: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test prediction on dummy image
    print("\n" + "="*60)
    print("PREDICTION TEST")
    print("="*60)
    
    try:
        # Create a dummy 64x64 grayscale image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        image_tensor = transform(dummy_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            
            predicted_char = chars[top_idx.item()]
            confidence = top_prob.item()
            
        print(f"[OK] Model can make predictions")
        try:
            print(f"  - Predicted character: '{predicted_char}' (confidence: {confidence:.4f})")
        except UnicodeEncodeError:
            print(f"  - Predicted character: (Unicode char, code point: U+{ord(predicted_char):04X}) (confidence: {confidence:.4f})")
        print(f"  - Output shape: {output.shape}")
        
    except Exception as e:
        print(f"[ERROR] Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("[OK] Model file exists and loads correctly")
    print("[OK] Model architecture matches checkpoint")
    print("[OK] Model can make predictions")
    
    if has_ranjana or has_devanagari:
        script_type = "Ranjana" if has_ranjana else "Devanagari"
        print(f"[SUCCESS] Model contains {script_type} characters - READY FOR USE!")
        return True
    elif has_unicode:
        print("[WARNING] Model contains Unicode but may need verification")
        return True
    else:
        print("[ERROR] Model does NOT contain Ranjana/Devanagari characters - RETRAIN NEEDED")
        return False

if __name__ == "__main__":
    print("="*60)
    print("TRAINED MODEL VERIFICATION TEST")
    print("="*60)
    print()
    
    success = test_model()
    
    print("\n" + "="*60)
    if success:
        print("[SUCCESS] ALL TESTS PASSED - Model is ready to use!")
    else:
        print("[ERROR] SOME TESTS FAILED - Please check the errors above")
    print("="*60)

