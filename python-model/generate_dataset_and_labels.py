import os
from PIL import Image, ImageDraw, ImageFont
import random
import sys

# -------------------------------
# Paths & parameters
# -------------------------------
font_path = r"E:\Cllz\FYP\fonts\NithyaRanjanaDU-Regular.otf"
corpus_file = "corpus.txt"
dataset_folder = "dataset"
images_folder = os.path.join(dataset_folder, "images")
labels_file = os.path.join(dataset_folder, "labels.txt")
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--variations', type=int, default=10)
args = ap.parse_args()
variations_per_line = args.variations  # control dataset size

img_height = 32  # CRNN expected height
img_width = 256  # flexible width, can adjust if needed

# -------------------------------
# Check font
# -------------------------------
try:
    font = ImageFont.truetype(font_path, 32)
    test_img = Image.new("L", (img_width, img_height), color=255)
    draw = ImageDraw.Draw(test_img)
    draw.text((0, 0), "नेपाली भाषा", font=font, fill=0)
    test_img.show()
    print("Font loaded successfully. Close test image to continue.")
except OSError:
    print("Cannot load font. Check the font file and path.")
    sys.exit(1)

# -------------------------------
# Prepare folders
# -------------------------------
os.makedirs(images_folder, exist_ok=True)

# -------------------------------
# Load corpus
# -------------------------------
with open(corpus_file, "r", encoding="utf-8") as f:
    corpus_lines = [line.strip() for line in f if line.strip()]

# -------------------------------
# Generate images & labels
# -------------------------------
labels = []
count = 0
for line in corpus_lines:
    for i in range(variations_per_line):
        # Create grayscale image with fixed height
        img = Image.new("L", (img_width, img_height), color=255)
        draw = ImageDraw.Draw(img)

        # Random offsets for variation
        x_offset = random.randint(0, 10)
        y_offset = random.randint(0, 5)

        draw.text((x_offset, y_offset), line, font=font, fill=0)

        img_name = f"img_{count:05d}.png"
        img.save(os.path.join(images_folder, img_name))
        labels.append(f"{img_name}|{line}")

        print(f"Saved {img_name} -> {line}")
        count += 1

# -------------------------------
# Save labels.txt
# -------------------------------
with open(labels_file, "w", encoding="utf-8") as f:
    f.write("\n".join(labels))

print(f"\nDataset generation complete! {count} images created.")
print(f"labels.txt saved at {labels_file}")
