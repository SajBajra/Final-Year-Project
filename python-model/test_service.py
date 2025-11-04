"""
Test script for ocr_service_ar.py
Verifies routes and basic functionality
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        import flask
        print("  [OK] Flask")
        
        import flask_cors
        print("  [OK] Flask-CORS")
        
        import torch
        print("  [OK] PyTorch")
        
        import cv2
        print("  [OK] OpenCV")
        
        import PIL
        print("  [OK] Pillow")
        
        print("[OK] All imports successful!\n")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}\n")
        return False

def test_model_file():
    """Test if model file exists"""
    print("Testing model file...")
    model_path = "best_character_crnn.pth"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        print(f"  [OK] Model file found: {model_path} ({size:.2f} MB)\n")
        return True
    else:
        print(f"  [FAIL] Model file not found: {model_path}")
        print("  [WARN] Service will start but model won't be loaded\n")
        return False

def test_routes():
    """Test if routes are defined correctly"""
    print("Testing route definitions...")
    try:
        # Read the service file
        with open("ocr_service_ar.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        routes = {
            "@app.route('/',": "Root route",
            "@app.route('/health',": "Health route",
            "@app.route('/predict',": "Predict route",
            "@app.route('/predict/base64',": "Base64 predict route"
        }
        
        all_found = True
        for route_pattern, route_name in routes.items():
            if route_pattern in content:
                print(f"  [OK] {route_name}")
            else:
                print(f"  [FAIL] {route_name} - NOT FOUND")
                all_found = False
        
        if all_found:
            print("[OK] All routes defined correctly!\n")
        else:
            print("[FAIL] Some routes missing!\n")
        
        return all_found
    except Exception as e:
        print(f"  ✗ Error reading file: {e}\n")
        return False

def test_cors():
    """Test if CORS is enabled"""
    print("Testing CORS configuration...")
    try:
        with open("ocr_service_ar.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        if "from flask_cors import CORS" in content and "CORS(app)" in content:
            print("  [OK] CORS is enabled\n")
            return True
        else:
            print("  [FAIL] CORS not properly configured\n")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}\n")
        return False

def test_model_class():
    """Test if CharacterCRNN class is defined"""
    print("Testing model class...")
    try:
        with open("ocr_service_ar.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        if "class CharacterCRNN" in content:
            print("  [OK] CharacterCRNN class defined")
            if "def forward(self, x):" in content:
                print("  [OK] Forward method defined")
                print("[OK] Model class structure correct!\n")
                return True
            else:
                print("  [FAIL] Forward method missing\n")
                return False
        else:
            print("  [FAIL] CharacterCRNN class not found\n")
            return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}\n")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Lipika OCR Service - Test Suite")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Model File", test_model_file()))
    results.append(("Routes", test_routes()))
    results.append(("CORS", test_cors()))
    results.append(("Model Class", test_model_class()))
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} - {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Service should work correctly.")
        return 0
    else:
        print("[WARNING] Some tests failed. Service may have issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
