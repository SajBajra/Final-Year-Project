# Issue: Model Trained with Devanagari but Predicting ASCII

## Problem
The model checkpoint (`best_character_crnn_improved.pth`) contains **ONLY Devanagari characters** (62 Unicode characters, 0 ASCII), which means the model WAS trained correctly with Devanagari labels.

However, the OCR service is still predicting ASCII characters like 'a' instead of Devanagari characters like 'अ'.

## Root Cause
The model was trained correctly with Devanagari labels, but something is causing predictions to be ASCII. There are two possibilities:

1. **The OCR service is loading a different/old checkpoint** that has ASCII characters
2. **There's a bug in how predictions are mapped** - the model predicts the correct index, but the character lookup is wrong

## Solution
Since the model checkpoint has Devanagari characters, we should NOT need the transliteration mapping anymore. The OCR service should directly output Devanagari characters.

However, the current mapping code is still helpful as a fallback until we verify the predictions are working correctly.

## Next Steps
1. Verify which model file the OCR service is actually loading
2. Check the logs when OCR service starts - it should show the character set
3. Remove the ASCII-to-Devanagari mapping once we confirm predictions are Devanagari

## Current Status
✅ Model checkpoint has Devanagari characters
✅ Dataset was prepared with Devanagari labels (via transliteration)
❌ OCR service predictions are still ASCII (needs investigation)

