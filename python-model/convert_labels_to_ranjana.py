"""
Convert English transliteration labels to Ranjana characters
This fixes the mismatch between dataset labels and model predictions
"""

import os
from transliteration_to_ranjana import transliterate_to_ranjana

def convert_labels_file(input_file, output_file):
    """Convert labels in a file from English to Ranjana"""
    
    converted_count = 0
    unchanged_count = 0
    errors = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                f_out.write('\n')
                continue
            
            # Parse line: filename|label
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 2:
                    filename = parts[0].strip()
                    english_label = parts[1].strip()
                    
                    # Convert to Ranjana
                    ranjana_label = transliterate_to_ranjana(english_label)
                    
                    # Write converted line
                    f_out.write(f"{filename}|{ranjana_label}\n")
                    
                    if ranjana_label != english_label:
                        converted_count += 1
                    else:
                        unchanged_count += 1
                        
                    # Track errors (if conversion failed)
                    if ranjana_label == english_label and english_label and not english_label.isascii():
                        # This shouldn't happen if conversion works
                        pass
                else:
                    errors.append((line_num, line))
                    f_out.write(line + '\n')
            else:
                # Invalid format
                errors.append((line_num, line))
                f_out.write(line + '\n')
    
    return converted_count, unchanged_count, errors

if __name__ == '__main__':
    import sys
    
    # Default paths
    base_dir = '../prepared_dataset'
    train_input = os.path.join(base_dir, 'train_labels.txt')
    val_input = os.path.join(base_dir, 'val_labels.txt')
    train_output = os.path.join(base_dir, 'train_labels_ranjana.txt')
    val_output = os.path.join(base_dir, 'val_labels_ranjana.txt')
    
    print("=" * 60)
    print("Converting Labels from English to Ranjana")
    print("=" * 60)
    
    # Convert training labels
    if os.path.exists(train_input):
        print(f"\nConverting training labels...")
        print(f"  Input:  {train_input}")
        print(f"  Output: {train_output}")
        
        conv, unchanged, errs = convert_labels_file(train_input, train_output)
        
        print(f"  Converted: {conv} labels")
        print(f"  Unchanged: {unchanged} labels")
        if errs:
            print(f"  Errors: {len(errs)} lines")
            for line_num, line in errs[:5]:
                print(f"    Line {line_num}: {line}")
    else:
        print(f"\n[ERROR] Training labels not found: {train_input}")
    
    # Convert validation labels
    if os.path.exists(val_input):
        print(f"\nConverting validation labels...")
        print(f"  Input:  {val_input}")
        print(f"  Output: {val_output}")
        
        conv, unchanged, errs = convert_labels_file(val_input, val_output)
        
        print(f"  Converted: {conv} labels")
        print(f"  Unchanged: {unchanged} labels")
        if errs:
            print(f"  Errors: {len(errs)} lines")
            for line_num, line in errs[:5]:
                print(f"    Line {line_num}: {line}")
    else:
        print(f"\n[ERROR] Validation labels not found: {val_input}")
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Backup original labels (optional):")
    print(f"     mv {train_input} {train_input}.backup")
    print(f"     mv {val_input} {val_input}.backup")
    print("\n  2. Use converted labels for training:")
    print(f"     python train_character_crnn_improved.py \\")
    print(f"       --images {base_dir}/images \\")
    print(f"       --train_labels {train_output} \\")
    print(f"       --val_labels {val_output} \\")
    print(f"       --epochs 150")
    print("\n  3. After training, the model will predict Ranjana correctly!")
