"""
Analyze Dataset Statistics
"""
import os
from pathlib import Path

def analyze_dataset(dataset_path='../Dataset'):
    """Analyze dataset statistics"""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Get all character folders
    char_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"Character classes: {len(char_folders)}")
    print()
    
    # Count images per character
    folder_stats = []
    total_images = 0
    
    for char_folder in char_folders:
        char_label = char_folder.name
        image_files = list(char_folder.glob('*.png')) + list(char_folder.glob('*.jpg')) + \
                     list(char_folder.glob('*.jpeg')) + list(char_folder.glob('*.PNG')) + \
                     list(char_folder.glob('*.JPG')) + list(char_folder.glob('*.JPEG'))
        image_count = len(image_files)
        total_images += image_count
        folder_stats.append((char_label, image_count))
    
    # Sort by image count
    folder_stats.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate statistics
    image_counts = [count for _, count in folder_stats]
    avg_images = sum(image_counts) / len(image_counts) if image_counts else 0
    min_images = min(image_counts) if image_counts else 0
    max_images = max(image_counts) if image_counts else 0
    median_images = sorted(image_counts)[len(image_counts)//2] if image_counts else 0
    
    print(f"Total images: {total_images}")
    print(f"Average images per character: {avg_images:.1f}")
    print(f"Minimum images per character: {min_images}")
    print(f"Maximum images per character: {max_images}")
    print(f"Median images per character: {median_images}")
    print()
    
    # Show top and bottom characters
    print("Top 10 characters by image count:")
    for i, (char, count) in enumerate(folder_stats[:10], 1):
        print(f"  {i:2d}. {char:15s}: {count:4d} images")
    print()
    
    print("Bottom 10 characters by image count:")
    for i, (char, count) in enumerate(folder_stats[-10:], 1):
        print(f"  {i:2d}. {char:15s}: {count:4d} images")
    print()
    
    # Characters with few images
    few_images = [(char, count) for char, count in folder_stats if count < 50]
    if few_images:
        print(f"Characters with < 50 images: {len(few_images)}")
        for char, count in few_images[:10]:
            print(f"  - {char}: {count} images")
        print()
    
    # Ideal dataset size calculation
    print("=" * 70)
    print("IDEAL DATASET SIZE ANALYSIS")
    print("=" * 70)
    
    ideal_min = 500
    ideal_recommended = 2000
    ideal_optimal = 5000
    
    current_total = total_images
    ideal_min_total = len(char_folders) * ideal_min
    ideal_recommended_total = len(char_folders) * ideal_recommended
    ideal_optimal_total = len(char_folders) * ideal_optimal
    
    print(f"Current dataset: {current_total:,} images")
    print(f"Minimum ideal (500/char): {ideal_min_total:,} images")
    print(f"Recommended (2000/char): {ideal_recommended_total:,} images")
    print(f"Optimal (5000/char): {ideal_optimal_total:,} images")
    print()
    
    print("Gap analysis:")
    print(f"  Current vs Minimum: {ideal_min_total - current_total:,} images needed ({((ideal_min_total - current_total) / ideal_min_total * 100):.1f}% of minimum)")
    print(f"  Current vs Recommended: {ideal_recommended_total - current_total:,} images needed ({((ideal_recommended_total - current_total) / ideal_recommended_total * 100):.1f}% of recommended)")
    print(f"  Current vs Optimal: {ideal_optimal_total - current_total:,} images needed ({((ideal_optimal_total - current_total) / ideal_optimal_total * 100):.1f}% of optimal)")
    print()
    
    # Training recommendations
    print("=" * 70)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 70)
    
    if current_total < ideal_min_total:
        print(f"[WARNING] Dataset is SMALL ({(current_total / ideal_min_total * 100):.1f}% of minimum)")
        print("   Recommendation: Expand dataset to at least 500 images per character")
        print("   Expected accuracy: 99.0-99.5% (character-level)")
        print("   Real-world performance: 85-90%")
        print("   Training epochs: 300-500")
    elif current_total < ideal_recommended_total:
        print(f"[OK] Dataset is MODERATE ({(current_total / ideal_recommended_total * 100):.1f}% of recommended)")
        print("   Recommendation: Expand dataset to 2000 images per character")
        print("   Expected accuracy: 99.5-99.8% (character-level)")
        print("   Real-world performance: 90-95%")
        print("   Training epochs: 200-300")
    else:
        print(f"[SUCCESS] Dataset is GOOD ({(current_total / ideal_optimal_total * 100):.1f}% of optimal)")
        print("   Recommendation: Continue training with current dataset")
        print("   Expected accuracy: 99.8-99.9% (character-level)")
        print("   Real-world performance: 95-99%")
        print("   Training epochs: 150-250")

if __name__ == '__main__':
    analyze_dataset()

