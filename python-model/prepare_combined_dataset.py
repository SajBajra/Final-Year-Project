"""
Prepare Combined Dataset for Training
Combines all datasets (Dataset, char_dataset) into one unified training dataset
Excludes user_char_datasets and user_datasets
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import random

def prepare_combined_dataset(
    dataset_folders=None,
    output_folder='../prepared_dataset',
    train_split=0.8,
    exclude_folders=None
):
    """
    Combine multiple dataset folders into one unified training dataset
    
    Args:
        dataset_folders: List of dataset folder paths to combine
        output_folder: Output folder for prepared dataset
        train_split: Train/validation split ratio
        exclude_folders: List of folder names to exclude
    """
    
    if dataset_folders is None:
        # Default: combine Dataset and char_dataset
        dataset_folders = ['../Dataset', '../char_dataset']
    
    if exclude_folders is None:
        exclude_folders = ['user_char_datasets', 'user_datasets']
    
    output_path = Path(output_folder)
    images_path = output_path / 'images'
    
    # Create output directories
    images_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all images with their character labels from all datasets
    all_data = []
    
    print("=" * 70)
    print("COMBINING ALL DATASETS")
    print("=" * 70)
    
    for dataset_folder in dataset_folders:
        dataset_path = Path(dataset_folder)
        
        if not dataset_path.exists():
            print(f"[WARN] Dataset folder not found: {dataset_path}")
            continue
        
        print(f"\nScanning dataset folder: {dataset_path}")
        
        # Handle two different structures:
        # 1. Character folders (Dataset/)
        # 2. Images folder with labels file (char_dataset/)
        
        if (dataset_path / 'images').exists() and (dataset_path / 'labels.txt').exists():
            # Structure: images/ folder + labels.txt
            print(f"  Found images/ + labels.txt structure")
            images_folder = dataset_path / 'images'
            labels_file = dataset_path / 'labels.txt'
            
            # Read labels file
            with open(labels_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            for line in lines:
                parts = line.split('|')
                if len(parts) == 2:
                    img_name, char_label = parts[0].strip(), parts[1].strip()
                    img_path = images_folder / img_name
                    if img_path.exists():
                        all_data.append((img_path, char_label))
            
            print(f"  Loaded {len([d for d in all_data if str(d[0]).startswith(str(dataset_path))])} images from labels file")
        
        else:
            # Structure: character folders
            char_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name not in exclude_folders])
            
            print(f"  Found {len(char_folders)} character folders")
            
            for char_folder in char_folders:
                char_label = char_folder.name
                
                # Get all image files in this folder
                image_files = list(char_folder.glob('*.png')) + list(char_folder.glob('*.jpg')) + \
                             list(char_folder.glob('*.jpeg')) + list(char_folder.glob('*.PNG')) + \
                             list(char_folder.glob('*.JPG')) + list(char_folder.glob('*.JPEG'))
                
                if len(image_files) > 0:
                    print(f"    {char_label}: {len(image_files)} images")
                
                for img_file in image_files:
                    all_data.append((img_file, char_label))
    
    print(f"\n{'='*70}")
    print(f"Total images collected: {len(all_data)}")
    
    if len(all_data) == 0:
        print("[ERROR] No images found in any dataset folder!")
        return
    
    # Shuffle data
    random.seed(42)
    random.shuffle(all_data)
    
    # Split into train and validation
    split_idx = int(len(all_data) * train_split)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Train: {len(train_data)} images ({len(train_data)/len(all_data)*100:.1f}%)")
    print(f"Val: {len(val_data)} images ({len(val_data)/len(all_data)*100:.1f}%)")
    
    # Get unique characters
    all_chars = sorted(set([label for _, label in all_data]))
    print(f"Character classes: {len(all_chars)}")
    
    # Copy images and create labels
    train_labels = []
    val_labels = []
    
    print(f"\n{'='*70}")
    print("Copying images to unified dataset...")
    print(f"{'='*70}")
    
    # Copy training images
    print(f"\nCopying {len(train_data)} training images...")
    for idx, (img_file, char_label) in enumerate(train_data):
        new_name = f"char_{idx+1:06d}.png"
        dest_path = images_path / new_name
        
        try:
            # Convert to PNG and ensure grayscale
            img = Image.open(img_file)
            if img.mode != 'L':
                img = img.convert('L')
            img.save(dest_path, 'PNG')
            
            train_labels.append(f"{new_name}|{char_label}")
            
            if (idx + 1) % 1000 == 0:
                print(f"  Copied {idx + 1}/{len(train_data)} training images...")
        except Exception as e:
            print(f"  Error copying {img_file}: {e}")
    
    # Copy validation images
    print(f"\nCopying {len(val_data)} validation images...")
    for idx, (img_file, char_label) in enumerate(val_data):
        new_name = f"char_{len(train_data) + idx + 1:06d}.png"
        dest_path = images_path / new_name
        
        try:
            img = Image.open(img_file)
            if img.mode != 'L':
                img = img.convert('L')
            img.save(dest_path, 'PNG')
            
            val_labels.append(f"{new_name}|{char_label}")
            
            if (idx + 1) % 1000 == 0:
                print(f"  Copied {idx + 1}/{len(val_data)} validation images...")
        except Exception as e:
            print(f"  Error copying {img_file}: {e}")
    
    # Write label files
    train_labels_file = output_path / 'train_labels.txt'
    val_labels_file = output_path / 'val_labels.txt'
    
    print(f"\nWriting label files...")
    with open(train_labels_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_labels))
    
    with open(val_labels_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_labels))
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"DATASET PREPARATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total images: {len(all_data)}")
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Character classes: {len(all_chars)}")
    print(f"\nCharacters found:")
    for i, char in enumerate(all_chars):
        if i < 20 or i >= len(all_chars) - 5:
            count = sum(1 for _, label in all_data if label == char)
            print(f"  {char}: {count} images")
        elif i == 20:
            print(f"  ... ({len(all_chars) - 25} more characters)")
    
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"\nDataset sources combined:")
    for folder in dataset_folders:
        print(f"  - {folder}")
    print(f"\nExcluded folders: {exclude_folders}")
    print(f"\n{'='*70}")
    print(f"\nNext step: Convert labels to Ranjana and train:")
    print(f"  python convert_labels_to_ranjana.py")
    print(f"  python train_character_crnn_improved.py \\")
    print(f"    --images {output_path / 'images'} \\")
    print(f"    --train_labels {output_path / 'train_labels_ranjana.txt'} \\")
    print(f"    --val_labels {output_path / 'val_labels_ranjana.txt'} \\")
    print(f"    --epochs 200 \\")
    print(f"    --batch_size 64")
    print(f"{'='*70}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Combined Dataset from Multiple Sources')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['../Dataset', '../char_dataset'],
                        help='List of dataset folder paths to combine')
    parser.add_argument('--output', type=str, default='../prepared_dataset',
                        help='Output folder for prepared dataset')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--exclude', type=str, nargs='+',
                        default=['user_char_datasets', 'user_datasets'],
                        help='Folder names to exclude')
    
    args = parser.parse_args()
    
    prepare_combined_dataset(
        dataset_folders=args.datasets,
        output_folder=args.output,
        train_split=args.train_split,
        exclude_folders=args.exclude
    )

