"""
Training script to train model for 500 epochs
Resumes from latest checkpoint if available
Ensures dataset images are recognized correctly
"""

import os
import sys
import subprocess

def main():
    print("=" * 70)
    print("LIPIKA - TRAINING TO 500 EPOCHS")
    print("=" * 70)
    print()
    print("This script will:")
    print("  1. Train the model for 500 epochs")
    print("  2. Automatically resume from latest checkpoint if available")
    print("  3. Save checkpoints every 5 epochs")
    print("  4. Ensure dataset images are recognized correctly")
    print()
    print("Goal: Achieve 99.3-99.6% validation accuracy")
    print("      Ensure dataset images are recognized with high confidence")
    print()
    
    # Check if prepared dataset exists
    if not os.path.exists('../prepared_dataset'):
        print("[WARN] Prepared dataset not found. Preparing dataset...")
        print("[INFO] Run: python prepare_combined_dataset.py")
        print("       OR: python train_all_datasets.py --prepare_dataset --epochs 500 --resume_latest")
        response = input("Do you want to prepare the dataset now? (y/n): ")
        if response.lower() == 'y':
            print("[INFO] Preparing dataset...")
            try:
                from prepare_combined_dataset import prepare_combined_dataset
                prepare_combined_dataset(
                    dataset_folders=['../Dataset'],
                    output_folder='../prepared_dataset',
                    train_split=0.8,
                    exclude_folders=['user_char_datasets', 'user_datasets']
                )
                print("[OK] Dataset prepared!")
            except Exception as e:
                print(f"[ERROR] Dataset preparation failed: {e}")
                sys.exit(1)
        else:
            print("[INFO] Please prepare the dataset first")
            sys.exit(1)
    
    # Check if Ranjana labels exist
    if not os.path.exists('../prepared_dataset/train_labels_ranjana.txt'):
        print("[WARN] Ranjana labels not found. Converting labels...")
        print("[INFO] Run: python convert_labels_to_ranjana.py")
        response = input("Do you want to convert labels now? (y/n): ")
        if response.lower() == 'y':
            print("[INFO] Converting labels to Ranjana...")
            try:
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
                print("[OK] Labels converted!")
            except Exception as e:
                print(f"[ERROR] Label conversion failed: {e}")
                sys.exit(1)
        else:
            print("[INFO] Please convert labels first")
            sys.exit(1)
    
    # Start training
    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()
    
    # Run training with resume from latest checkpoint
    cmd = [
        sys.executable,
        'train_all_datasets.py',
        '--epochs', '500',
        '--batch_size', '64',
        '--lr', '0.001',
        '--checkpoint_interval', '5',
        '--resume_latest'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    print("Training will:")
    print("  - Resume from latest checkpoint (if available)")
    print("  - Train for 500 epochs total")
    print("  - Save checkpoints every 5 epochs")
    print("  - Use batch size 64, learning rate 0.001")
    print()
    print("Estimated time:")
    print("  CPU: ~10 hours (300 more epochs from checkpoint)")
    print("  GPU: ~2.5 hours (300 more epochs from checkpoint)")
    print()
    
    response = input("Start training? (y/n): ")
    if response.lower() != 'y':
        print("[INFO] Training cancelled")
        sys.exit(0)
    
    # Run training
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode == 0:
            print()
            print("=" * 70)
            print("TRAINING COMPLETE!")
            print("=" * 70)
            print()
            print("Next steps:")
            print("  1. Check best_character_crnn_improved.pth for the best model")
            print("  2. Restart OCR service to use the new model")
            print("  3. Test with dataset images to verify recognition")
            print("  4. Check validation accuracy (should be 99.3-99.6%)")
            print()
        else:
            print(f"[ERROR] Training failed with exit code {result.returncode}")
            sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("[INFO] Training interrupted by user")
        print("[INFO] You can resume training later with: --resume_latest")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

