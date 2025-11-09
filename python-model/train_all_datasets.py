"""
Complete Training Script for All Combined Datasets
Trains model on all available datasets (Dataset + char_dataset)
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Model on All Combined Datasets')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--prepare_dataset', action='store_true',
                        help='Prepare combined dataset before training')
    parser.add_argument('--convert_labels', action='store_true',
                        help='Convert labels to Ranjana before training')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help='Save periodic checkpoint every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIPIKA - COMPLETE MODEL TRAINING ON ALL DATASETS")
    print("=" * 70)
    print()
    
    # Step 1: Prepare combined dataset if requested
    if args.prepare_dataset:
        print("[STEP 1/4] Preparing combined dataset...")
        print("-" * 70)
        try:
            from prepare_combined_dataset import prepare_combined_dataset
            prepare_combined_dataset(
                dataset_folders=['../Dataset'],  # Only use Dataset folder
                output_folder='../prepared_dataset',
                train_split=0.8,
                exclude_folders=['user_char_datasets', 'user_datasets']
            )
            print("[OK] Dataset preparation complete!")
        except Exception as e:
            print(f"[ERROR] Dataset preparation failed: {e}")
            sys.exit(1)
        print()
    else:
        print("[STEP 1/4] Skipping dataset preparation (use --prepare_dataset to enable)")
        print()
    
    # Step 2: Convert labels to Ranjana if requested
    if args.convert_labels:
        print("[STEP 2/4] Converting labels to Ranjana...")
        print("-" * 70)
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, 'convert_labels_to_ranjana.py'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"[ERROR] Label conversion failed: {result.stderr}")
                sys.exit(1)
            print("[OK] Label conversion complete!")
        except Exception as e:
            print(f"[ERROR] Label conversion failed: {e}")
            sys.exit(1)
        print()
    else:
        print("[STEP 2/4] Skipping label conversion (use --convert_labels to enable)")
        print("[INFO] Make sure train_labels_ranjana.txt and val_labels_ranjana.txt exist")
        print()
    
    # Step 3: Verify dataset files exist
    print("[STEP 3/4] Verifying dataset files...")
    print("-" * 70)
    
    images_path = '../prepared_dataset/images'
    train_labels = '../prepared_dataset/train_labels_ranjana.txt'
    val_labels = '../prepared_dataset/val_labels_ranjana.txt'
    
    # Fallback to non-Ranjana labels if Ranjana don't exist
    if not os.path.exists(train_labels):
        train_labels = '../prepared_dataset/train_labels.txt'
        print("[WARN] Ranjana labels not found, using original labels")
        print("[INFO] Run convert_labels_to_ranjana.py to convert to Ranjana")
    
    if not os.path.exists(val_labels):
        val_labels = '../prepared_dataset/val_labels.txt'
    
    if not os.path.exists(images_path):
        print(f"[ERROR] Images folder not found: {images_path}")
        print("[INFO] Run with --prepare_dataset to create dataset")
        sys.exit(1)
    
    if not os.path.exists(train_labels):
        print(f"[ERROR] Training labels not found: {train_labels}")
        sys.exit(1)
    
    if not os.path.exists(val_labels):
        print(f"[ERROR] Validation labels not found: {val_labels}")
        sys.exit(1)
    
    # Count images
    image_count = len([f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"[OK] Found {image_count} images")
    print(f"[OK] Training labels: {train_labels}")
    print(f"[OK] Validation labels: {val_labels}")
    print()
    
    # Step 4: Train model
    print("[STEP 4/4] Starting model training...")
    print("-" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Checkpoint interval: Every {args.checkpoint_interval} epochs")
    print()
    print("This will train the model for the full number of epochs (no early stopping)")
    print(f"Checkpoints will be saved every {args.checkpoint_interval} epochs to: checkpoints/")
    print("Estimated time:")
    print(f"  CPU: ~{args.epochs * 0.02:.1f} hours")
    print(f"  GPU: ~{args.epochs * 0.005:.1f} hours")
    print()
    
    try:
        from train_character_crnn_improved import train_improved_model
        
        best_acc = train_improved_model(
            images_folder=images_path,
            train_labels=train_labels,
            val_labels=val_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume_from=None,
            checkpoint_interval=args.checkpoint_interval
        )
        
        print()
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"[SUCCESS] Model trained on all combined datasets")
        print(f"[ACCURACY] Best validation accuracy: {best_acc:.2f}%")
        print(f"[FILE] Model saved as: best_character_crnn_improved.pth")
        print()
        print("Next steps:")
        print("  1. Restart OCR service to use new model")
        print("  2. Test with Ranjana script images")
        print("  3. Verify character recognition accuracy")
        print("=" * 70)
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

