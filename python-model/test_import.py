"""
Quick test script to check if PyTorch/torchvision imports work correctly
Run this before starting the OCR service to diagnose import issues
"""

import sys
import time

print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print("-" * 60)

try:
    print("\n[1/3] Importing torch...")
    start = time.time()
    import torch
    elapsed = time.time() - start
    print(f"✓ PyTorch {torch.__version__} imported in {elapsed:.2f} seconds")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    
    print("\n[2/3] Importing torchvision...")
    start = time.time()
    from torchvision import transforms
    elapsed = time.time() - start
    print(f"✓ torchvision imported in {elapsed:.2f} seconds")
    
    print("\n[3/3] Testing transforms...")
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
    ])
    print("✓ Transforms working correctly")
    
    print("\n" + "=" * 60)
    print("✓ ALL IMPORTS SUCCESSFUL!")
    print("=" * 60)
    print("\nYou can now run: python ocr_service_ar.py")
    
except KeyboardInterrupt:
    print("\n" + "=" * 60)
    print("✗ IMPORT WAS INTERRUPTED (KeyboardInterrupt)")
    print("=" * 60)
    print("\nPossible causes:")
    print("1. You pressed Ctrl+C - wait for imports to complete")
    print("2. Import is taking too long - try using Python 3.11 or 3.12")
    print("3. Package installation is corrupted - reinstall PyTorch")
    print("\nSee TROUBLESHOOT_PYTHON313.md for solutions")
    sys.exit(1)
    
except ImportError as e:
    print("\n" + "=" * 60)
    print("✗ IMPORT ERROR")
    print("=" * 60)
    print(f"\nError: {e}")
    print("\nPossible solutions:")
    print("1. Reinstall packages: pip install -r requirements.txt")
    print("2. Use Python 3.11 or 3.12 instead of 3.13")
    print("3. Check TROUBLESHOOT_PYTHON313.md for detailed help")
    sys.exit(1)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("✗ UNEXPECTED ERROR")
    print("=" * 60)
    print(f"\nError: {type(e).__name__}: {e}")
    print("\nSee TROUBLESHOOT_PYTHON313.md for troubleshooting")
    sys.exit(1)

