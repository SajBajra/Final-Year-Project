# GitHub Push Guide - Clean History

## Current Status
✅ Code cleaned and ready  
✅ Character-based AR OCR ready  
❌ Git history contains 165K+ files (large models/images)  
⚠️  Need to clean history before push

## Problem
Your Git history has:
- 89 model checkpoint files (*.pth) - ~50GB total
- 1,000+ training dataset images  
- 164,000+ character dataset images
- Total repository size: **VERY LARGE** (>100GB likely)

## Solution: Clean Git History

### Option 1: Force Push (SIMPLEST) ⭐ **RECOMMENDED**

This will rewrite GitHub history but remove all large files:

```bash
# Push current clean state
git push origin main --force

# WARNING: This overwrites GitHub history!
# Only do this if you're okay with losing old commits
```

### Option 2: Fresh Repository (SAFEST)

Start with a clean repo:

```bash
# Remove .git folder
Remove-Item -Recurse -Force .git

# Reinitialize
git init
git add .
git commit -m "Initial commit: Character-based AR OCR system"

# Add GitHub remote
git remote add origin <your-github-repo-url>

# Push
git push -u origin main
```

### Option 3: git filter-branch (Complex)

Use git filter-branch to remove large files from history:

```bash
# Remove all .pth files from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch *.pth" \
  --prune-empty --tag-name-filter cat -- --all

# Remove dataset images
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch dataset/images/*.png" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now

# Force push
git push origin main --force
```

## What Gets Pushed

### ✅ Will Be Pushed
```
FYP/
├── python-model/           # OCR service + training
│   ├── train_character_crnn.py
│   ├── ocr_service_ar.py
│   ├── app.py
│   ├── cli.py
│   ├── requirements.txt
│   └── README_CHARACTER.md
├── javabackend/            # Java backend (skeleton)
├── frontend/               # React frontend (skeleton)
├── char_dataset/           # Character images (164K images)
│   ├── images/
│   └── train_labels.txt
├── README.md
├── SETUP.md
├── .gitignore
└── Documentation files
```

### ❌ Won't Be Pushed (Gitignored)
- `*.pth` - Model files
- `checkpoints/` - Training checkpoints
- `dataset/images/*.png` - Training images
- `fonts/` - Large font files
- Generated images/logs

## ⚠️ IMPORTANT: Character Dataset Size

**Problem**: `char_dataset/images/` has 164,000 images  
**Status**: NOT in .gitignore yet!  
**Impact**: Will try to push ~20-30GB of images  

### Fix Before Pushing

Add to `.gitignore`:
```bash
# Add character dataset images to .gitignore
echo "char_dataset/images/" >> .gitignore
```

## Recommended Push Steps

### Step 1: Update .gitignore
```bash
# Make sure char_dataset images are ignored
notepad .gitignore  # Or use any editor
# Add: char_dataset/images/
```

### Step 2: Remove Large Dataset from Git
```bash
# Remove char_dataset from being tracked
git rm -r --cached char_dataset/

# Commit the removal
git commit -m "Stop tracking large char_dataset images"
```

### Step 3: Choose Push Method

**If you control the repo (single developer):**
```bash
git push origin main --force
```

**If you share the repo:**
Use Option 2 (Fresh Repository) to avoid breaking others' clones

### Step 4: Verify Push Worked
```bash
# Check GitHub repo size
# Should be < 100MB, not > 100GB
```

## After Push

### Train Model
```bash
cd python-model
python train_character_crnn.py \
  --images ../char_dataset/images \
  --train_labels ../char_dataset/train_labels.txt \
  --val_labels ../char_dataset/val_labels.txt \
  --epochs 100
```

### Use Model
```bash
# Start AR OCR service
python ocr_service_ar.py

# Upload image to http://localhost:5000
# Get bounding boxes for AR overlay
```

---

**Choose your option and proceed!**

