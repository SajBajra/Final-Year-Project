# Python-Model Folder - Useless Files Analysis

## Summary
This document identifies files that are not needed for the production application.

---

## 🔴 DEFINITELY USELESS (Can be deleted)

### 1. **ocr_service.py** ❌
- **Why useless**: This is the OLD OCR service without AR features
- **Replacement**: `ocr_service_ar.py` is the active service (provides character bounding boxes)
- **Evidence**: Backend expects character bbox data, which only `ocr_service_ar.py` provides
- **Status**: Can be safely deleted

### 2. **app.py** ❌
- **Why useless**: Legacy Flask app wrapper that uses `ocr_service.py` (which is also useless)
- **Note**: Says "Prefer using ocr_service.py for production" but that's outdated
- **Status**: Can be safely deleted

### 3. **checkpoints/** folder (100+ files) ❌
- **Why useless**: Contains 100+ intermediate training checkpoint files (epoch_0005.pth through epoch_0500.pth)
- **Keep**: Only `best_model.pth` and `final_model.pth` if they're the best models
- **Delete**: All epoch_*.pth files (intermediate checkpoints)
- **Size**: These take up significant space

### 4. **Test/Debug Scripts** ❌
- `test_import.py` - Just tests imports, not needed in production
- `test_service.py` - Tests ocr_service.py (which is useless)
- `test_integration.py` - Integration tests, not needed in production
- `test_trained_model.py` - Model testing script
- `test_model_on_training_data.py` - Training data testing
- `diagnose_model_issue.py` - Diagnostic script
- `check_which_model.py` - Model checking utility
- `check_model_chars.py` - Character checking utility

### 5. **Training Scripts** ❌ (if not actively training)
- `train.py` - Old training script
- `train_character_crnn_improved.py` - Training script (only needed for retraining)
- `analyze_dataset.py` - Dataset analysis tool
- `prepare_dataset.py` - Dataset preparation
- `prepare_combined_dataset.py` - Combined dataset prep
- `generate_dataset_and_labels.py` - Dataset generation
- `convert_labels_to_ranjana.py` - Label conversion utility
- `transliteration_to_ranjana.py` - Transliteration utility

### 6. **CLI Tool** ❌
- `cli.py` - Command-line interface (not used in production)

### 7. **Templates** ❌ (if not using web interface)
- `templates/` folder - HTML templates for web interface
- **Note**: Only needed if running standalone web app (not used with Java backend)

### 8. **Training Results** ❌
- `training_curves.png` - Training visualization
- `training_curves_improved.png` - Training visualization
- `model_test_results.json` - Test results

### 9. **Character Lists** ❌
- `chars.txt` - Character list (NOT USED - chars are loaded from model checkpoint)
- `enhanced_chars.txt` - Enhanced character list (NOT USED - chars are loaded from model checkpoint)
- **Note**: Characters are embedded in the .pth model files, so these .txt files are not needed

---

## ✅ KEEP (Essential for production)

### 1. **ocr_service_ar.py** ✅
- **Active service**: This is the one being used
- **Features**: Provides character bounding boxes for AR overlay
- **Status**: MUST KEEP

### 2. **requirements.txt** ✅
- **Essential**: Lists all dependencies
- **Status**: MUST KEEP

### 3. **Model Files** ✅ (only the best ones)
- `best_character_crnn_improved.pth` - **ACTIVE** (loaded first, preferred)
- `character_crnn_improved_final.pth` - **FALLBACK** (loaded if first not found)
- `best_character_crnn.pth` - **FALLBACK** (loaded if improved models not found)
- **Note**: ocr_service_ar.py tries these in order. Keep all 3 for safety, or just the first one if it works.

### 4. **Character Lists** ❌
- **NOT NEEDED**: Characters are loaded from model checkpoint files (.pth), not from .txt files
- `chars.txt` and `enhanced_chars.txt` can be deleted

---

## 📊 File Size Impact

### Large files to consider:
- **checkpoints/** folder: ~100+ files, potentially several GB
- **Model files (.pth)**: Each can be 10-100MB
- **Training scripts**: Small but numerous

### Estimated space savings:
- Deleting checkpoints folder: **Potentially 1-5 GB**
- Deleting old OCR service: **~50-100 KB**
- Deleting test scripts: **~50-100 KB**
- Deleting training scripts: **~100-200 KB**

---

## 🎯 Recommended Action Plan

1. **Immediate deletion** (safe):
   - `ocr_service.py`
   - `app.py`
   - All test scripts
   - All training scripts (if not retraining)
   - Templates folder (if not using standalone web app)
   - Training visualization files

2. **Check before deletion**:
   - Which model file is actually loaded? (check ocr_service_ar.py)
   - Which chars.txt is used? (check ocr_service_ar.py)
   - Keep only the active model and char list

3. **Checkpoints cleanup**:
   - Keep only `best_model.pth` and `final_model.pth` if they're the best
   - Delete all `epoch_*.pth` files
   - This will save the most space

---

## 🔍 How to Verify Active Files

Run these commands to check what's actually used:

```python
# Check which model file is loaded
grep -n "load.*\.pth" ocr_service_ar.py

# Check which chars file is loaded  
grep -n "chars.*\.txt" ocr_service_ar.py
```

