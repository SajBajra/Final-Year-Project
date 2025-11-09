"""
Lipika - IMPROVED Character-Based Ranjana OCR Training
Enhanced with data augmentation, better preprocessing, and improved architecture
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
import unicodedata
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np

# Try to import tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️ tqdm not available")

# -------------------
# IMPROVED Data Augmentation
# -------------------
class AdvancedAugmentation:
    """Advanced data augmentation for better generalization"""
    
    def __init__(self):
        pass
    
    def random_rotate(self, img, max_angle=15):
        """Random rotation within ±max_angle degrees"""
        angle = random.uniform(-max_angle, max_angle)
        try:
            # Try with fillcolor first (newer Pillow versions)
            return img.rotate(angle, fillcolor=255, resample=Image.BILINEAR)
        except TypeError:
            # Fallback for older versions - expand=True adds white borders
            return img.rotate(angle, expand=True, resample=Image.BILINEAR)
    
    def random_affine(self, img):
        """Random affine transformation"""
        # Small translations, rotations, scaling
        translate_x = random.uniform(-3, 3)
        translate_y = random.uniform(-3, 3)
        angle = random.uniform(-5, 5)
        scale = random.uniform(0.9, 1.1)
        
        try:
            # Try with fillcolor first (newer versions)
            return TF.affine(
                img,
                angle=angle,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=random.uniform(-2, 2),
                fillcolor=255
            )
        except TypeError:
            # Fallback for older versions without fillcolor
            return TF.affine(
                img,
                angle=angle,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=random.uniform(-2, 2)
            )
    
    def random_noise(self, img, noise_factor=0.05):
        """Add random noise"""
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, noise_factor * 255, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))
    
    def random_blur(self, img, max_radius=1):
        """Random blur"""
        if random.random() < 0.3:
            radius = random.uniform(0, max_radius)
            if radius > 0:
                return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
    
    def random_brightness(self, img, factor_range=(0.8, 1.2)):
        """Random brightness adjustment"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    
    def random_contrast(self, img, factor_range=(0.8, 1.2)):
        """Random contrast adjustment"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    def random_erosion_dilation(self, img):
        """Morphological operations to simulate printing variations"""
        if not HAS_CV2:
            return img  # Skip if cv2 not available
        
        img_array = np.array(img)
        
        if random.random() < 0.3:
            # Erosion (make text thinner)
            kernel = np.ones((2, 2), np.uint8)
            img_array = cv2.erode(img_array, kernel, iterations=1)
        elif random.random() < 0.3:
            # Dilation (make text thicker)
            kernel = np.ones((2, 2), np.uint8)
            img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        return Image.fromarray(img_array)
    
    def random_elastic_transform(self, img):
        """Elastic deformation for realistic variations"""
        if not HAS_SCIPY:
            return img  # Skip if scipy not available
        
        img_array = np.array(img)
        
        if random.random() < 0.2:  # Apply 20% of the time
            try:
                alpha = random.uniform(50, 150)
                sigma = random.uniform(5, 10)
                
                shape = img_array.shape
                dx = gaussian_filter((random.random() * 2 - 1), sigma) * alpha
                dy = gaussian_filter((random.random() * 2 - 1), sigma) * alpha
                
                x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
                indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
                
                img_array = map_coordinates(img_array, indices, order=1, mode='constant').reshape(shape)
                return Image.fromarray(img_array.astype(np.uint8))
            except:
                return img  # Return original if elastic transform fails
        
        return img
    
    def __call__(self, img):
        """Apply random augmentation"""
        if random.random() < 0.7:  # 70% chance to augment
            # Random rotation
            if random.random() < 0.5:
                img = self.random_rotate(img)
            
            # Random affine
            if random.random() < 0.5:
                img = self.random_affine(img)
            
            # Random brightness/contrast
            if random.random() < 0.4:
                img = self.random_brightness(img)
            if random.random() < 0.4:
                img = self.random_contrast(img)
            
            # Random noise
            if random.random() < 0.3:
                img = self.random_noise(img)
            
            # Random blur
            if random.random() < 0.2:
                img = self.random_blur(img)
        
        return img

# Try to import scipy and cv2 for advanced augmentations
try:
    from scipy.ndimage import gaussian_filter, map_coordinates
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_filter = None
    map_coordinates = None
    print("⚠️ scipy not available, elastic transform disabled")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    cv2 = None
    print("⚠️ cv2 not available, morphological operations disabled")

# -------------------
# IMPROVED Character CRNN Model
# -------------------
class ImprovedCharacterCRNN(nn.Module):
    """
    Improved Character-based CRNN with better architecture
    - Deeper CNN with residual connections
    - Attention mechanism
    - Better regularization
    """
    def __init__(self, num_classes, img_height=64, img_width=64, dropout=0.5):
        super(ImprovedCharacterCRNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # Improved CNN Feature Extractor with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.conv1_res = nn.Conv2d(1, 32, 1)  # Skip connection
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Sigmoid()
        )
        
        # Improved RNN
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if dropout > 0 else 0
        )
        
        # Improved Classification head with more layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # First conv with residual
        out = self.conv1(x) + self.conv1_res(x)
        
        # Continue through CNN
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        
        # Global pooling
        conv_features = self.global_pool(out)  # [B, 256, 1, 1]
        conv_features = conv_features.squeeze(-1).squeeze(-1)  # [B, 256]
        
        # Apply attention
        attention_weights = self.attention(conv_features)  # [B, 256]
        conv_features = conv_features * attention_weights
        
        # Reshape for RNN: [B, C] -> [B, 1, C]
        conv_features = conv_features.unsqueeze(1)
        
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
# IMPROVED Character Dataset with Augmentation
# -------------------
class ImprovedCharacterDataset(Dataset):
    def __init__(self, images_folder, labels_file, transform=None, augment=True):
        self.images_folder = images_folder
        self.transform = transform
        self.augment = augment
        self.augmentation = AdvancedAugmentation() if augment else None
        
        # Read labels
        with open(labels_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Parse labels
        self.data = []
        for line in lines:
            parts = line.split("|")
            if len(parts) == 2:
                img_name, label = parts[0].strip(), parts[1].strip()
                if label:
                    self.data.append([img_name, normalize_unicode(label)])
        
        print(f"Loaded {len(self.data)} character samples from {labels_file}")
        
        # Get character set
        labels = [item[1] for item in self.data]
        chars_set = set("".join(labels))
        self.chars = sorted(list(chars_set))
        
        print(f"Character set size: {len(self.chars)} characters")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.images_folder, img_name)
        
        try:
            img = Image.open(img_path).convert("L")
            
            # Apply augmentation if training
            if self.augment and self.augmentation:
                img = self.augmentation(img)
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            img = Image.new("L", (64, 64), 255)
        
        # Convert character to index
        char_idx = self.chars.index(label) if label in self.chars else 0
        
        if self.transform:
            img = self.transform(img)
        
        return img, char_idx

# -------------------
# IMPROVED Training Function
# -------------------
def train_improved_model(images_folder, train_labels, val_labels, epochs=150, batch_size=64, learning_rate=0.001, resume_from=None, checkpoint_interval=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoints directory for periodic saves
    ckpt_dir = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    print(f"Periodic checkpoints every {checkpoint_interval} epochs")
    
    # IMPROVED Transforms with better normalization
    train_transform = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Better normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Load datasets WITH augmentation
    train_dataset = ImprovedCharacterDataset(
        images_folder, train_labels, 
        transform=train_transform, 
        augment=True  # Enable augmentation for training
    )
    val_dataset = ImprovedCharacterDataset(
        images_folder, val_labels, 
        transform=val_transform, 
        augment=False  # No augmentation for validation
    )
    
    # Create dataloaders with more workers and better prefetching
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    
    # Create IMPROVED model
    num_classes = len(train_dataset.chars)
    model = ImprovedCharacterCRNN(num_classes=num_classes, img_height=64, img_width=64, dropout=0.5).to(device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accs = []
    val_acc_history = []
    val_loss_history = []
    last_best_epoch = 0
    
    if resume_from and os.path.exists(resume_from):
        try:
            print(f"\n[INFO] Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_acc = checkpoint.get('val_acc', 0.0)
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            val_accs = checkpoint.get('val_accs', [])
            
            # Initialize history from checkpoint
            val_acc_history = val_accs.copy() if val_accs else []
            val_loss_history = val_losses.copy() if val_losses else []
            
            # Find last best epoch
            if val_accs:
                last_best_epoch = start_epoch - 1
                for i, acc in enumerate(val_accs):
                    if acc >= best_val_acc - 0.0001:  # Allow small floating point differences
                        last_best_epoch = i
            
            print(f"[OK] Resumed from epoch {start_epoch}")
            print(f"[OK] Best val_acc: {best_val_acc:.4f}% (epoch {last_best_epoch+1})")
            print(f"[OK] Loaded {len(val_accs)} previous epochs of history")
        except Exception as e:
            print(f"[ERROR] Failed to resume from checkpoint: {e}")
            print("[INFO] Starting training from scratch...")
            import traceback
            traceback.print_exc()
    
    print(f"\nImproved Model created with {num_classes} character classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # IMPROVED Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # IMPROVED Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Load optimizer state if resuming
    if resume_from and os.path.exists(resume_from):
        try:
            checkpoint = torch.load(resume_from, map_location=device)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("[OK] Loaded optimizer state")
        except Exception as e:
            print(f"[WARN] Could not load optimizer state: {e}")
    
    # IMPROVED Learning rate scheduler - Combine CosineAnnealing with ReduceLROnPlateau
    # CosineAnnealing for cyclical learning rate
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7  # Increased T_0 for slower cycles
    )
    
    # ReduceLROnPlateau for plateau detection (will be called manually)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=False, min_lr=1e-7
    )
    
    # Training loop
    if start_epoch == 0:
        print(f"\nStarting IMPROVED training for {epochs} epochs (full training, no early stopping)...\n")
    else:
        print(f"\nContinuing IMPROVED training from epoch {start_epoch} to {epochs} epochs (full training, no early stopping)...\n")
    
    # Track improvement trends
    improvement_window = 10  # Track improvements over last N epochs
    # Calculate epochs since improvement from checkpoint
    if val_accs and len(val_accs) > 0:
        epochs_since_improvement = len(val_accs) - last_best_epoch - 1
    else:
        epochs_since_improvement = 0
    
    for epoch in range(start_epoch, epochs):
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Update learning rate schedulers
        cosine_scheduler.step()  # Cosine annealing (cyclical)
        old_lr = optimizer.param_groups[0]['lr']
        plateau_scheduler.step(val_acc)  # Reduce LR on plateau (based on val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"  [🔄] Learning rate reduced: {old_lr:.8f} -> {new_lr:.8f} (plateau detected)")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_acc_history.append(val_acc)
        val_loss_history.append(val_loss)
        
        # Calculate improvement trends
        improvement_info = ""
        trend_info = ""
        if len(val_acc_history) >= improvement_window:
            # Calculate average over last N epochs
            recent_avg = sum(val_acc_history[-improvement_window:]) / improvement_window
            older_avg = sum(val_acc_history[-improvement_window*2:-improvement_window]) / improvement_window if len(val_acc_history) >= improvement_window*2 else recent_avg
            
            # Calculate trend
            trend = recent_avg - older_avg
            if trend > 0.01:
                trend_info = f"📈 Trending UP (+{trend:.3f}% over last {improvement_window} epochs)"
            elif trend < -0.01:
                trend_info = f"📉 Trending DOWN ({trend:.3f}% over last {improvement_window} epochs)"
            else:
                trend_info = f"➡️  Stable (change: {trend:.3f}% over last {improvement_window} epochs)"
        
        # Check for improvement (even small ones)
        improvement = val_acc - best_val_acc
        if improvement > 0:
            epochs_since_improvement = 0
            last_best_epoch = epoch
            if improvement >= 0.01:
                improvement_info = f"✨ IMPROVED by +{improvement:.4f}%"
            else:
                improvement_info = f"✨ Small improvement: +{improvement:.4f}%"
        else:
            epochs_since_improvement += 1
            improvement_info = f"⏸️  No improvement ({epochs_since_improvement} epochs)"
        
        # Check if validation loss improved (even if accuracy didn't)
        loss_improvement = ""
        if len(val_loss_history) >= 2:
            loss_change = val_loss_history[-2] - val_loss  # Positive = improvement
            if loss_change > 0.0001:
                loss_improvement = f" (Loss improved: -{loss_change:.6f})"
            elif loss_change < -0.0001:
                loss_improvement = f" (Loss increased: +{abs(loss_change):.6f})"
        
        # Print progress with detailed information
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}%")
        print(f"  Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.8f}")
        if improvement_info:
            print(f"  {improvement_info}{loss_improvement}")
        if trend_info:
            print(f"  {trend_info}")
        if len(val_acc_history) >= improvement_window:
            recent_min = min(val_acc_history[-improvement_window:])
            recent_max = max(val_acc_history[-improvement_window:])
            print(f"  Recent range: {recent_min:.4f}% - {recent_max:.4f}% (last {improvement_window} epochs)")
        print()
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'chars': train_dataset.chars,
            'num_classes': num_classes,
            'model_type': 'ImprovedCharacterCRNN',
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs
        }
        
        # Save best model (even for tiny improvements)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            improvement_amount = val_acc - (val_accs[-2] if len(val_accs) > 1 else val_acc)
            # Save best model to root directory
            torch.save(checkpoint_data, 'best_character_crnn_improved.pth')
            # Also save to checkpoints directory
            best_ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"  [✅] Saved best model: {val_acc:.4f}% (improvement: +{improvement_amount:.4f}%)")
            print(f"  [📊] Best accuracy: {best_val_acc:.4f}% (epoch {epoch+1})\n")
        else:
            improvement_needed = best_val_acc - val_acc
            print(f"  [📌] Best: {best_val_acc:.4f}% (epoch {last_best_epoch+1}) | Current: {val_acc:.4f}% | Gap: {improvement_needed:.4f}%")
            print(f"  [⏱️] {epochs_since_improvement} epochs since last improvement\n")
        
        # Periodic checkpoint every N epochs
        if (epoch + 1) % checkpoint_interval == 0:
            periodic_path = os.path.join(ckpt_dir, f'epoch_{epoch+1:04d}.pth')
            torch.save(checkpoint_data, periodic_path)
            print(f"  [OK] Saved periodic checkpoint: epoch_{epoch+1:04d}.pth\n")
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.4f}% (epoch {last_best_epoch+1})")
    print(f"Final validation accuracy: {val_accs[-1]:.4f}%")
    print(f"Total improvement: {best_val_acc - (val_accs[0] if val_accs else best_val_acc):.4f}%")
    
    # Calculate statistics
    if len(val_accs) > 0:
        print(f"\nAccuracy Statistics:")
        print(f"  Initial: {val_accs[0]:.4f}%")
        print(f"  Final: {val_accs[-1]:.4f}%")
        print(f"  Best: {best_val_acc:.4f}%")
        print(f"  Average: {sum(val_accs) / len(val_accs):.4f}%")
        print(f"  Last 10 epochs avg: {sum(val_accs[-10:]) / min(10, len(val_accs)):.4f}%")
        if len(val_accs) >= 20:
            print(f"  Last 20 epochs avg: {sum(val_accs[-20:]) / 20:.4f}%")
    
    if len(val_losses) > 0:
        print(f"\nLoss Statistics:")
        print(f"  Initial: {val_losses[0]:.6f}")
        print(f"  Final: {val_losses[-1]:.6f}")
        print(f"  Best: {min(val_losses):.6f}")
        print(f"  Improvement: {val_losses[0] - val_losses[-1]:.6f}")
    
    print(f"{'='*70}\n")
    
    # Save final model
    final_checkpoint = {
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'train_loss': train_loss,
        'chars': train_dataset.chars,
        'num_classes': num_classes,
        'model_type': 'ImprovedCharacterCRNN',
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    final_path = os.path.join(ckpt_dir, 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    torch.save(final_checkpoint, 'character_crnn_improved_final.pth')
    print(f"[OK] Saved final model: {final_path}")
    print(f"[OK] Saved final model: character_crnn_improved_final.pth")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accs, label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Plot learning rate (need to track it during training)
    # For now, show validation accuracy trend
    if len(val_accs) >= 20:
        # Show moving average
        window = 10
        moving_avg = [sum(val_accs[max(0, i-window):i+1]) / min(i+1, window) for i in range(len(val_accs))]
        plt.plot(val_accs, label='Val Accuracy', color='green', alpha=0.3)
        plt.plot(moving_avg, label=f'Moving Avg ({window})', color='darkgreen', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy Trend')
        plt.legend()
    else:
        plt.plot(val_accs, label='Val Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')
        plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_improved.png', dpi=150)
    print("Training curves saved to training_curves_improved.png")
    
    return best_val_acc

# Note: This file is now a module. Use train.py as the main entry point.
# train.py handles dataset preparation, label conversion, and calls train_improved_model()
