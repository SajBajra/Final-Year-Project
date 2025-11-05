"""
Retrain Model with Ranjana Characters
This script retrains the character recognition model using Ranjana Unicode labels
instead of ASCII transliteration labels.
"""

import os
import sys
import argparse
from train_character_crnn_improved import train_improved_model

def main():
    parser = argparse.ArgumentParser(description='Retrain Character CRNN Model with Ranjana Labels')
    parser.add_argument('--images', type=str, default='../prepared_dataset/images',
                        help='Path to character images folder')
    parser.add_argument('--train_labels', type=str, default='../prepared_dataset/train_labels_ranjana.txt',
                        help='Path to training labels file (Ranjana)')
    parser.add_argument('--val_labels', type=str, default='../prepared_dataset/val_labels_ranjana.txt',
                        help='Path to validation labels file (Ranjana)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--backup', action='store_true',
                        help='Backup existing model before training')
    
    args = parser.parse_args()
    
    # Check if label files exist
    if not os.path.exists(args.train_labels):
        print(f"[ERROR] Training labels not found: {args.train_labels}")
        print(f"[INFO] Make sure you have run convert_labels_to_ranjana.py first")
        sys.exit(1)
    
    if not os.path.exists(args.val_labels):
        print(f"[ERROR] Validation labels not found: {args.val_labels}")
        print(f"[INFO] Make sure you have run convert_labels_to_ranjana.py first")
        sys.exit(1)
    
    if not os.path.exists(args.images):
        print(f"[ERROR] Images folder not found: {args.images}")
        sys.exit(1)
    
    # Backup existing model if requested
    if args.backup:
        model_path = 'best_character_crnn_improved.pth'
        if os.path.exists(model_path):
            backup_path = f'{model_path}.backup'
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"[INFO] Backed up existing model to: {backup_path}")
    
    print("=" * 70)
    print("LIPIKA - RETRAINING MODEL WITH RANJANA CHARACTERS")
    print("=" * 70)
    print(f"Images folder: {args.images}")
    print(f"Train labels: {args.train_labels} (Ranjana Unicode)")
    print(f"Val labels: {args.val_labels} (Ranjana Unicode)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70)
    print("\n[INFO] This will train a new model that predicts Ranjana Unicode characters")
    print("[INFO] The old model will be replaced with the new trained model\n")
    
    # Verify labels contain Ranjana characters
    print("[INFO] Verifying labels contain Ranjana characters...")
    with open(args.train_labels, 'r', encoding='utf-8') as f:
        sample_lines = [f.readline().strip() for _ in range(10)]
    
    ranjana_count = 0
    for line in sample_lines:
        if '|' in line:
            label = line.split('|')[1].strip()
            # Check if label contains Unicode characters (Ranjana)
            if any(ord(c) > 127 for c in label):
                ranjana_count += 1
    
    if ranjana_count == 0:
        print(f"[WARN] No Ranjana characters found in sample labels!")
        print(f"[WARN] Make sure train_labels_ranjana.txt contains Ranjana Unicode characters")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("[INFO] Aborted.")
            sys.exit(0)
    else:
        print(f"[OK] Found Ranjana characters in labels ({ranjana_count}/10 samples)")
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    
    # Train the model
    best_acc = train_improved_model(
        images_folder=args.images,
        train_labels=args.train_labels,
        val_labels=args.val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_from=None  # Start fresh training
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"[SUCCESS] Model trained with Ranjana characters")
    print(f"[ACCURACY] Best validation accuracy: {best_acc:.2f}%")
    print(f"[FILE] Model saved as: best_character_crnn_improved.pth")
    print("\n[INFO] You can now restart the OCR service to use the new model")
    print("[INFO] The model will now predict Ranjana Unicode characters correctly!")

if __name__ == '__main__':
    main()

