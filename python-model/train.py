"""
Lipika - Complete Training Script for Ranjana OCR
Combines all improvements: dataset preparation, label conversion, and training
Default: 500 epochs with all enhancements
"""

import os
import sys
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(
        description='Train Ranjana OCR Model - Complete Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch (500 epochs, auto-prepare dataset)
  python train.py
  
  # Train with custom epochs
  python train.py --epochs 300
  
  # Resume from latest checkpoint
  python train.py --resume_latest
  
  # Resume from specific checkpoint
  python train.py --resume checkpoints/epoch_0200.pth
        """
    )
    
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint (e.g., checkpoints/epoch_0200.pth)')
    parser.add_argument('--resume_latest', action='store_true',
                        help='Automatically resume from latest checkpoint')
    parser.add_argument('--auto_setup', action='store_true', default=True,
                        help='Automatically check and setup dataset/labels if needed (default: True)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LIPIKA - COMPLETE RANJANA OCR TRAINING")
    print("=" * 70)
    print(f"Default epochs: 500 (customize with --epochs)")
    print(f"All improvements included: dual schedulers, progress tracking, etc.")
    print("=" * 70)
    print()
    
    # Step 1: Prepare dataset if needed
    if args.prepare_dataset or (args.auto_setup and not os.path.exists('../prepared_dataset/images')):
        print("[STEP 1/3] Preparing dataset...")
        print("-" * 70)
        try:
            from prepare_combined_dataset import prepare_combined_dataset
            prepare_combined_dataset(
                dataset_folders=['../Dataset'],
                output_folder='../prepared_dataset',
                train_split=0.8,
                exclude_folders=['user_char_datasets', 'user_datasets']
            )
            print("[OK] Dataset preparation complete!")
        except Exception as e:
            print(f"[ERROR] Dataset preparation failed: {e}")
            if not args.prepare_dataset:
                print("[INFO] Dataset already exists or preparation failed. Continuing...")
            else:
                sys.exit(1)
        print()
    else:
        print("[STEP 1/3] Dataset check...")
        if os.path.exists('../prepared_dataset/images'):
            image_count = len([f for f in os.listdir('../prepared_dataset/images') 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"[OK] Dataset found: {image_count} images")
        else:
            print("[WARN] Dataset not found. Run with --prepare_dataset to create it.")
        print()
    
    # Step 2: Convert labels if needed
    if args.convert_labels or (args.auto_setup and not os.path.exists('../prepared_dataset/train_labels_ranjana.txt')):
        print("[STEP 2/3] Converting labels to Ranjana...")
        print("-" * 70)
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, 'convert_labels_to_ranjana.py'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("[OK] Label conversion complete!")
            else:
                print(f"[WARN] Label conversion returned non-zero: {result.stderr}")
                if not args.convert_labels:
                    print("[INFO] Labels may already be converted. Continuing...")
        except Exception as e:
            print(f"[WARN] Label conversion failed: {e}")
            if not args.convert_labels:
                print("[INFO] Labels may already be converted. Continuing...")
        print()
    else:
        print("[STEP 2/3] Labels check...")
        if os.path.exists('../prepared_dataset/train_labels_ranjana.txt'):
            print("[OK] Ranjana labels found")
        else:
            print("[WARN] Ranjana labels not found. Run with --convert_labels to convert them.")
        print()
    
    # Step 3: Verify dataset files
    print("[STEP 3/3] Verifying dataset files...")
    print("-" * 70)
    
    images_path = '../prepared_dataset/images'
    train_labels = '../prepared_dataset/train_labels_ranjana.txt'
    val_labels = '../prepared_dataset/val_labels_ranjana.txt'
    
    # Fallback to non-Ranjana labels if Ranjana don't exist
    if not os.path.exists(train_labels):
        train_labels = '../prepared_dataset/train_labels.txt'
        print("[WARN] Ranjana labels not found, using original labels")
        print("[INFO] Run: python convert_labels_to_ranjana.py to convert to Ranjana")
    
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
    
    # Handle resume from checkpoint
    resume_from = args.resume
    if args.resume_latest and not resume_from:
        # Find latest checkpoint
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        if os.path.exists(checkpoint_dir):
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
            if checkpoint_files:
                # Sort by epoch number
                checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
                resume_from = checkpoint_files[-1]
                print(f"[INFO] Found latest checkpoint: {os.path.basename(resume_from)}")
            else:
                # Try best model or final model
                best_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_character_crnn_improved.pth')
                final_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'character_crnn_improved_final.pth')
                if os.path.exists(final_model):
                    resume_from = final_model
                    print(f"[INFO] Found final model: {os.path.basename(resume_from)}")
                elif os.path.exists(best_model):
                    resume_from = best_model
                    print(f"[INFO] Found best model: {os.path.basename(resume_from)}")
    
    # Calculate remaining epochs for time estimation
    remaining = args.epochs
    if resume_from:
        print(f"[INFO] Resuming from checkpoint: {resume_from}")
        # Load checkpoint to get current epoch
        try:
            import torch
            checkpoint = torch.load(resume_from, map_location='cpu')
            current_epoch = checkpoint.get('epoch', 0)
            val_acc = checkpoint.get('val_acc', 0.0)
            print(f"[INFO] Checkpoint info: Epoch {current_epoch + 1}, Val Acc: {val_acc:.4f}%")
            remaining = max(0, args.epochs - (current_epoch + 1))
            if remaining <= 0:
                print(f"[WARN] Checkpoint is already at epoch {current_epoch + 1}, but target is {args.epochs}")
                print(f"[INFO] Will continue training from epoch {current_epoch + 1} to {args.epochs}")
                remaining = 0
        except Exception as e:
            print(f"[WARN] Could not read checkpoint info: {e}")
            print("[INFO] Will attempt to resume anyway")
    else:
        print("[INFO] Starting training from scratch")
    
    print()
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Checkpoint interval: Every {args.checkpoint_interval} epochs")
    print(f"Remaining epochs: {remaining}")
    print()
    print("Features:")
    print("  ✅ Dual learning rate schedulers (CosineAnnealing + ReduceLROnPlateau)")
    print("  ✅ Detailed progress tracking (improvements, trends, loss)")
    print("  ✅ Advanced data augmentation")
    print("  ✅ Improved architecture with attention")
    print("  ✅ Automatic checkpoint resume")
    print()
    print("Estimated time:")
    print(f"  CPU: ~{remaining * 0.02:.1f} hours ({remaining} epochs)")
    print(f"  GPU: ~{remaining * 0.005:.1f} hours ({remaining} epochs)")
    print("=" * 70)
    print()
    
    # Start training
    try:
        from train_character_crnn_improved import train_improved_model
        
        best_acc = train_improved_model(
            images_folder=images_path,
            train_labels=train_labels,
            val_labels=val_labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume_from=resume_from,
            checkpoint_interval=args.checkpoint_interval
        )
        
        print()
        print("=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"[SUCCESS] Model trained on all combined datasets")
        print(f"[ACCURACY] Best validation accuracy: {best_acc:.4f}%")
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

