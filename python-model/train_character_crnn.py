"""
Lipika - Character-Based Ranjana OCR Training
Trains a CRNN model for individual character recognition (Google Lens style)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import unicodedata
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# Try to import tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️ tqdm not available")

# -------------------
# Character CRNN Model (Optimized for single characters)
# -------------------
class CharacterCRNN(nn.Module):
    """
    Character-based CRNN model for individual character recognition
    Input: 64x64 grayscale character images
    Output: Character class probabilities
    """
    def __init__(self, num_classes, img_height=64, img_width=64):
        super(CharacterCRNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN Feature Extractor (optimized for single characters)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # Block 2
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # Block 3
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # Block 4
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            # Block 5 - Global pooling
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))  # 4x4 -> 1x1
        )
        
        # RNN for sequence modeling
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        conv_features = self.cnn(x)
        
        # Reshape for RNN: [B, C, 1, 1] -> [B, 1, C]
        b, c, h, w = conv_features.size()
        conv_features = conv_features.view(b, c, -1).permute(0, 2, 1)
        
        # RNN processing
        rnn_out, _ = self.rnn(conv_features)
        
        # Classification
        output = self.classifier(rnn_out.squeeze(1))
        
        return output

# -------------------
# Utility Functions
# -------------------
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

# -------------------
# Character Dataset
# -------------------
class CharacterDataset(Dataset):
    def __init__(self, images_folder, labels_file, transform=None, augment=True):
        self.images_folder = images_folder
        self.transform = transform
        self.augment = augment
        
        # Read labels
        with open(labels_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Parse labels (skip empty ones)
        self.data = []
        for line in lines:
            parts = line.split("|")
            if len(parts) == 2:
                img_name, label = parts[0].strip(), parts[1].strip()
                # Skip blank/empty labels
                if label:
                    self.data.append([img_name, normalize_unicode(label)])
        
        print(f"Loaded {len(self.data)} character samples from {labels_file}")
        
        # Get character set
        labels = [item[1] for item in self.data]
        chars_set = set("".join(labels))
        self.chars = sorted(list(chars_set))
        
        # Add blank token at index 0 for CTC
        if '' not in self.chars:
            self.chars = [''] + self.chars
        
        print(f"Character set size: {len(self.chars)} characters")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.images_folder, img_name)
        
        try:
            img = Image.open(img_path).convert("L")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new("L", (64, 64), 255)
        
        # Convert character to index
        char_idx = self.chars.index(label) if label in self.chars else 0
        
        if self.transform:
            img = self.transform(img)
        
        return img, char_idx

# -------------------
# Main Training Function
# -------------------
def train_character_model(images_folder, train_labels, val_labels, epochs=100, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    train_dataset = CharacterDataset(images_folder, train_labels, transform=train_transform)
    val_dataset = CharacterDataset(images_folder, val_labels, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    num_classes = len(train_dataset.chars)
    model = CharacterCRNN(num_classes=num_classes, img_height=64, img_width=64).to(device)
    
    print(f"\nModel created with {num_classes} character classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    
    print(f"\nStarting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]") if HAS_TQDM else train_loader
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]") if HAS_TQDM else val_loader
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'chars': train_dataset.chars,
                'num_classes': num_classes,
            }, 'best_character_crnn.pth')
            print(f"  ✓ Saved best model with val_acc: {val_acc:.2f}%\n")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Character-based CRNN for Ranjana OCR')
    parser.add_argument('--images', default='../char_dataset/images', help='Images folder')
    parser.add_argument('--train_labels', default='../char_dataset/train_labels.txt', help='Training labels file')
    parser.add_argument('--val_labels', default='../char_dataset/val_labels.txt', help='Validation labels file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    train_character_model(
        images_folder=args.images,
        train_labels=args.train_labels,
        val_labels=args.val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

