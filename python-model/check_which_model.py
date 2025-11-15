#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check which model files exist and their character sets
"""

import torch
import os
import sys

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

model_files = ['best_character_crnn.pth', 'best_character_crnn_improved.pth', 'character_crnn_improved_final.pth']

print("=" * 70)
print("MODEL FILE CHECK")
print("=" * 70)

for model_file in model_files:
    if os.path.exists(model_file):
        try:
            ckpt = torch.load(model_file, map_location='cpu')
            chars = ckpt.get('chars', [])
            model_type = ckpt.get('model_type', 'CharacterCRNN')
            
            ascii_chars = [c for c in chars if c and c.isascii() and (c.isalpha() or c.isdigit())]
            unicode_chars = [c for c in chars if c and ord(c) > 127]
            
            print(f"\n{model_file}:")
            print(f"  Exists: YES")
            print(f"  Model type: {model_type}")
            print(f"  Total characters: {len(chars)}")
            print(f"  ASCII characters: {len(ascii_chars)}")
            print(f"  Unicode (Devanagari) characters: {len(unicode_chars)}")
            if len(ascii_chars) > 0:
                print(f"  WARNING: Contains ASCII characters!")
                print(f"  ASCII chars: {ascii_chars[:10]}")
            if len(unicode_chars) > 0:
                print(f"  Devanagari chars (first 10): {unicode_chars[:10]}")
                
        except Exception as e:
            print(f"\n{model_file}:")
            print(f"  Exists: YES")
            print(f"  ERROR loading: {e}")
    else:
        print(f"\n{model_file}:")
        print(f"  Exists: NO")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("The OCR service loads models in this order:")
print("  1. best_character_crnn.pth (OLD - may have ASCII)")
print("  2. best_character_crnn_improved.pth (NEW - should have Devanagari)")
print("\nIf best_character_crnn.pth exists, it will be loaded FIRST!")
print("Delete it or change the loading order to use the improved model.")

