"""
Prepare Dataset for Training
Converts Dataset folder structure (character folders) to training format
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import random

def prepare_dataset(dataset_folder='../Dataset', output_folder='../prepared_dataset', train_split=0.8):
    """
    Convert Dataset folder structure to training format
    
    Dataset structure:
    Dataset/
      a/          (character folders)
        img1.png
        img2.png
      aa/
        img1.png
        ...
    
    Output structure:
    prepared_dataset/
      images/
        char_000001.png
        char_000002.png
        ...
      train_labels.txt
      val_labels.txt
    """
    
    dataset_path = Path(dataset_folder)
    output_path = Path(output_folder)
    images_path = output_path / 'images'
    
    # Create output directories
    images_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all images with their character labels
    all_data = []
    
    print(f"Scanning dataset folder: {dataset_path}")
    
    # Get all character folders
    char_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(char_folders)} character folders")
    
    for char_folder in char_folders:
        char_label = char_folder.name  # Character name from folder (e.g., 'a', 'aa', 'ba')
        
        # Get all image files in this folder
        image_files = list(char_folder.glob('*.png')) + list(char_folder.glob('*.jpg')) + \
                     list(char_folder.glob('*.jpeg')) + list(char_folder.glob('*.PNG')) + \
                     list(char_folder.glob('*.JPG')) + list(char_folder.glob('*.JPEG'))
        
        print(f"  {char_label}: {len(image_files)} images")
        
        for img_file in image_files:
            all_data.append((img_file, char_label))
    
    print(f"\nTotal images found: {len(all_data)}")
    
    # Shuffle data
    random.seed(42)
    random.shuffle(all_data)
    
    # Split into train and validation
    split_idx = int(len(all_data) * train_split)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Train: {len(train_data)} images")
    print(f"Val: {len(val_data)} images")
    
    # Copy images and create labels
    train_labels = []
    val_labels = []
    
    print("\nCopying training images...")
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
                print(f"  Copied {idx + 1}/{len(train_data)} images")
        except Exception as e:
            print(f"  Error copying {img_file}: {e}")
    
    print("\nCopying validation images...")
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
                print(f"  Copied {idx + 1}/{len(val_data)} images")
        except Exception as e:
            print(f"  Error copying {img_file}: {e}")
    
    # Write label files
    train_labels_file = output_path / 'train_labels.txt'
    val_labels_file = output_path / 'val_labels.txt'
    
    print(f"\nWriting {train_labels_file}...")
    with open(train_labels_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_labels))
    
    print(f"Writing {val_labels_file}...")
    with open(val_labels_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_labels))
    
    # Print character set statistics
    all_chars = sorted(set([label for _, label in all_data]))
    print(f"\n{'='*60}")
    print(f"Dataset Preparation Complete!")
    print(f"{'='*60}")
    print(f"Total images: {len(all_data)}")
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Character classes: {len(all_chars)}")
    print(f"\nCharacters found:")
    print(f"  {', '.join(all_chars[:20])}")
    if len(all_chars) > 20:
        print(f"  ... and {len(all_chars) - 20} more")
    print(f"\nOutput directory: {output_path.absolute()}")
    print(f"\nNow you can train with:")
    print(f"  python train_character_crnn_improved.py \\")
    print(f"    --images {output_path / 'images'} \\")
    print(f"    --train_labels {train_labels_file} \\")
    print(f"    --val_labels {val_labels_file} \\")
    print(f"    --epochs 150 \\")
    print(f"    --batch_size 64")
    print(f"{'='*60}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Dataset for Training')
    parser.add_argument('--dataset', type=str, default='../Dataset',
                        help='Path to Dataset folder with character subfolders')
    parser.add_argument('--output', type=str, default='../prepared_dataset',
                        help='Output folder for prepared dataset')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/validation split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    prepare_dataset(
        dataset_folder=args.dataset,
        output_folder=args.output,
        train_split=args.train_split
    )
