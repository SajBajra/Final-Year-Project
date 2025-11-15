#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check what character set is in the model checkpoint
"""

import torch
import sys
import os

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

model_path = "best_character_crnn_improved.pth"

if not os.path.exists(model_path):
    print(f"[ERROR] Model file not found: {model_path}")
    sys.exit(1)

print(f"[INFO] Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')

chars = checkpoint.get('chars', [])
print(f"\n[INFO] Total characters in model: {len(chars)}")

# Check character types
ascii_chars = []
unicode_chars = []

for c in chars:
    if c and isinstance(c, str):
        if c.isascii() and (c.isalpha() or c.isdigit()):
            ascii_chars.append(c)
        elif ord(c) > 127:
            unicode_chars.append(c)

print(f"\n[INFO] ASCII characters: {len(ascii_chars)}")
print(f"[INFO] Unicode (Devanagari) characters: {len(unicode_chars)}")

print(f"\n[INFO] First 20 characters in model:")
for i, c in enumerate(chars[:20]):
    char_type = "ASCII" if c in ascii_chars else "Unicode"
    unicode_code = f"U+{ord(c):04X}" if c else "N/A"
    print(f"  [{i:2d}] '{c}' ({char_type}, {unicode_code})")

print(f"\n[INFO] ASCII chars in model: {ascii_chars[:10] if ascii_chars else 'None'}...")
print(f"[INFO] Devanagari chars in model: {unicode_chars[:10] if unicode_chars else 'None'}...")

if len(ascii_chars) > 0:
    print(f"\n[WARN] Model contains ASCII characters - this may be from old training")
    print(f"[WARN] Model should contain ONLY Devanagari characters")
    
if len(unicode_chars) > 0 and len(ascii_chars) > 0:
    print(f"\n[INFO] Model has MIXED character set - both ASCII and Devanagari")
    print(f"[INFO] This explains why predictions are ASCII - model was trained with both!")
    
if len(unicode_chars) > 0 and len(ascii_chars) == 0:
    print(f"\n[OK] Model contains ONLY Devanagari characters")
    print(f"[WARN] But OCR service is predicting ASCII - check OCR service code!")

