import os
import unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

# Try to import tqdm, but fall back to simple progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️ tqdm not available, using simple progress display")

# -------------------
# Enhanced CRNN Model
# -------------------
class EnhancedCRNN(nn.Module):
    """
    Enhanced CRNN model with better architecture for higher accuracy
    """
    def __init__(self, num_classes, img_height=32, img_width=128):
        super(EnhancedCRNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # Enhanced CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1 - More filters for better feature extraction
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # Block 3 - Deeper layers
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 4 - More filters
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 5 - Additional layer for better features
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True),
        )
        
        # Enhanced RNN with more layers and dropout
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=3,  # More layers
            bidirectional=True,
            batch_first=True,
            dropout=0.2  # Higher dropout
        )
        
        # Character classification with dropout
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)  # [B, 512, 1, W]
        b, c, h, w = conv.size()
        assert h == 1, f"Height after CNN must be 1, got {h}"
        
        # Reshape for RNN: [B, W, C]
        conv = conv.squeeze(2)  # [B, 512, W]
        conv = conv.permute(0, 2, 1)  # [B, W, 512]
        
        # RNN processing
        recurrent, _ = self.rnn(conv)  # [B, W, 512]
        
        # Character classification
        output = self.fc(recurrent)  # [B, W, num_classes]
        
        # CTC expects [T, B, C] format
        output = output.permute(1, 0, 2)  # [W, B, num_classes]
        
        return output

# -------------------
# Utility Functions
# -------------------
def ctc_decode(output, chars, blank_idx=0):
    """Decode CRNN output using CTC decoding with proper blank handling"""
    import torch.nn.functional as F
    
    # Apply softmax to get probabilities
    probs = F.softmax(output, dim=2)
    
    # Get most likely characters
    argmax = probs.argmax(2)  # [T, batch]
    
    batch_size = argmax.size(1)
    decoded_strings = []
    
    for b in range(batch_size):
        sequence = argmax[:, b].cpu().numpy()
        
        # Remove consecutive duplicates and blanks
        decoded = []
        prev = -1
        
        for idx in sequence:
            if idx != prev and idx != blank_idx:
                if idx < len(chars):
                    decoded.append(chars[idx])
            prev = idx
        
        decoded_strings.append(''.join(decoded))
    
    return decoded_strings

def calculate_input_lengths(images, model=None):
    """Calculate input lengths for CTC loss based on actual model output"""
    batch_size = images.size(0)
    
    if model is not None:
        # Get actual output length by running a forward pass
        with torch.no_grad():
            output = model(images)
            actual_length = output.size(0)  # T dimension
    else:
        # Fallback calculation: CNN reduces width by factor of 4
        width = images.size(3)
        actual_length = width // 4
    
    return torch.full((batch_size,), actual_length, dtype=torch.long)

def ctc_loss_with_lengths(model_output, labels_concat, input_lengths, target_lengths, blank_idx=0):
    """Calculate CTC loss with proper length handling"""
    import torch.nn.functional as F
    
    # Apply log_softmax for CTC loss
    log_probs = F.log_softmax(model_output, dim=2)
    # Ensure float32 for CUDA ctc_loss (Half not supported)
    if log_probs.dtype != torch.float32:
        log_probs = log_probs.float()
    
    # Calculate CTC loss
    loss = F.ctc_loss(
        log_probs=log_probs,
        targets=labels_concat,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=blank_idx,
        reduction='mean',
        zero_infinity=True
    )
    
    return loss

def calculate_accuracy(predictions, targets):
    """Calculate character-level and word-level accuracy"""
    if len(predictions) != len(targets):
        return {"char_acc": 0.0, "word_acc": 0.0}
    
    char_correct = 0
    char_total = 0
    word_correct = 0
    
    for pred, target in zip(predictions, targets):
        # Character accuracy
        char_correct += sum(1 for p, t in zip(pred, target) if p == t)
        char_total += max(len(pred), len(target))
        
        # Word accuracy
        if pred == target:
            word_correct += 1
    
    char_acc = char_correct / char_total if char_total > 0 else 0.0
    word_acc = word_correct / len(predictions) if predictions else 0.0
    
    return {"char_acc": char_acc, "word_acc": word_acc}

def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate using Levenshtein distance."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    # DP edit distance
    dp = [[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1):
        dp[i][0] = i
    for j in range(len(hyp)+1):
        dp[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len(ref)][len(hyp)] / max(1, len(ref))

def compute_wer(ref: str, hyp: str) -> float:
    """Word Error Rate on whitespace-tokenized words."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dp = [[0]*(len(hyp_words)+1) for _ in range(len(ref_words)+1)]
    for i in range(len(ref_words)+1):
        dp[i][0] = i
    for j in range(len(hyp_words)+1):
        dp[0][j] = j
    for i in range(1, len(ref_words)+1):
        for j in range(1, len(hyp_words)+1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len(ref_words)][len(hyp_words)] / max(1, len(ref_words))

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)

# -------------------
# Enhanced Dataset with Augmentation
# -------------------
class EnhancedOCRDataset(Dataset):
    def __init__(self, images_folder, labels_file, transform=None, chars=None, augment=True):
        self.images_folder = images_folder
        self.transform = transform
        self.augment = augment

        # Read labels
        with open(labels_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        split_lines = [line.split("|") for line in lines]
        # Unicode normalize labels and validate
        self.data = []
        invalid_lines = []
        for i, l in enumerate(split_lines, 1):
            if len(l) != 2:
                invalid_lines.append((i, l))
                continue
            img, lab = l[0].strip(), normalize_unicode(l[1].strip())
            img_path = os.path.join(self.images_folder, img)
            if not os.path.exists(img_path):
                invalid_lines.append((i, l, f"Image not found: {img_path}"))
                continue
            self.data.append([img, lab])

        if invalid_lines:
            print(f"⚠️ Found {len(invalid_lines)} invalid lines:")
            for i, line_info in invalid_lines[:5]:  # Show up to 5 for brevity
                if len(line_info) == 2:
                    print(f"  Line {i}: {line_info} (Expected format: 'filename|label')")
                else:
                    print(f"  Line {i}: {line_info[0]} ({line_info[1]})")
            if len(self.data) == 0:
                raise ValueError("No valid data found after checking all lines!")

        if len(self.data) == 0:
            print(f"⚠️ Found {len(lines)} lines, but none were valid. Example(s): {split_lines[:5]}")
            raise ValueError("No valid data found!")

        self.img_files = [l[0] for l in self.data]
        self.labels = [l[1] for l in self.data]

        # Generate or use provided character set
        if chars is None:
            chars_set = set("".join(self.labels))
            self.chars = sorted(list(chars_set))
        else:
            self.chars = chars

        # Add blank token at index 0 for CTC
        if '' not in self.chars:
            self.chars = [''] + self.chars  # blank token at index 0

        print(f"Loaded {len(self.img_files)} images and {len(self.labels)} labels")
        print(f"Character set size: {len(self.chars)}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.img_files[idx])
        
        try:
            img = Image.open(img_path).convert("L")  # grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            img = Image.new("L", (128, 32), 255)
        
        label = normalize_unicode(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        # Map label characters to indices (skip blank token at index 0)
        label_tensor = torch.tensor([self.chars.index(c) for c in label if c in self.chars], dtype=torch.long)
        return img, label_tensor, len(label_tensor)

# -------------------
# Collate Function
# -------------------
def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, 0)
    labels_concat = torch.cat(labels)
    return images, labels_concat, torch.tensor(lengths, dtype=torch.long)

# -------------------
# Enhanced Training Functions
# -------------------
def train_epoch(model, dataloader, criterion, optimizer, device, chars):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Create progress bar or simple counter
    if HAS_TQDM:
        dataloader = tqdm(dataloader, desc="Training")
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    for batch_idx, (images, labels_concat, target_lengths) in enumerate(dataloader):
        images = images.to(device)
        labels_concat = labels_concat.to(device)
        target_lengths = target_lengths.to(device)
        
        # Calculate input lengths
        input_lengths = calculate_input_lengths(images, model).to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = model(images)
            # Calculate CTC loss
            loss = ctc_loss_with_lengths(outputs, labels_concat, input_lengths, target_lengths)
        scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
        # Decode predictions for monitoring (every 20 batches to save time)
        if batch_idx % 20 == 0:
            with torch.no_grad():
                predictions = ctc_decode(outputs, chars)
                # Get target strings for this batch
                batch_targets = []
                start_idx = 0
                for length in target_lengths:
                    end_idx = start_idx + length
                    target_indices = labels_concat[start_idx:end_idx].cpu().numpy()
                    target_str = ''.join([chars[i] for i in target_indices if i < len(chars)])
                    batch_targets.append(target_str)
                    start_idx = end_idx
                
                all_predictions.extend(predictions)
                all_targets.extend(batch_targets)
        
        # Simple progress display if no tqdm
        if not HAS_TQDM and batch_idx % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    # Calculate accuracy and error rates
    accuracy = calculate_accuracy(all_predictions, all_targets)
    if all_targets:
        cer_vals = [compute_cer(t, p) for p, t in zip(all_predictions, all_targets)]
        wer_vals = [compute_wer(t, p) for p, t in zip(all_predictions, all_targets)]
        accuracy["cer"] = float(sum(cer_vals) / len(cer_vals))
        accuracy["wer"] = float(sum(wer_vals) / len(wer_vals))
    
    return total_loss / len(dataloader), accuracy

def validate_epoch(model, dataloader, criterion, device, chars):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Create progress bar or simple counter
    if HAS_TQDM:
        dataloader = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, (images, labels_concat, target_lengths) in enumerate(dataloader):
            images = images.to(device)
            labels_concat = labels_concat.to(device)
            target_lengths = target_lengths.to(device)
            
            input_lengths = calculate_input_lengths(images, model).to(device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
            
            loss = ctc_loss_with_lengths(outputs, labels_concat, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Decode predictions
            predictions = ctc_decode(outputs, chars)
            
            # Get target strings
            batch_targets = []
            start_idx = 0
            for length in target_lengths:
                end_idx = start_idx + length
                target_indices = labels_concat[start_idx:end_idx].cpu().numpy()
                target_str = ''.join([chars[i] for i in target_indices if i < len(chars)])
                batch_targets.append(target_str)
                start_idx = end_idx
            
            all_predictions.extend(predictions)
            all_targets.extend(batch_targets)
            
            # Simple progress display if no tqdm
            if not HAS_TQDM and batch_idx % 10 == 0:
                print(f"  Val Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    accuracy = calculate_accuracy(all_predictions, all_targets)
    if all_targets:
        cer_vals = [compute_cer(t, p) for p, t in zip(all_predictions, all_targets)]
        wer_vals = [compute_wer(t, p) for p, t in zip(all_predictions, all_targets)]
        accuracy["cer"] = float(sum(cer_vals) / len(cer_vals))
        accuracy["wer"] = float(sum(wer_vals) / len(wer_vals))
    
    return total_loss / len(dataloader), accuracy

# -------------------
# Plot Metrics
# -------------------
def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot character accuracy
    plt.subplot(1, 2, 2)
    train_char_acc = [acc['char_acc'] for acc in train_accs]
    val_char_acc = [acc['char_acc'] for acc in val_accs]
    plt.plot(train_char_acc, label='Train Char Acc')
    plt.plot(val_char_acc, label='Val Char Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Character Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# -------------------
# Main Enhanced Training
# -------------------
if __name__ == "__main__":
    import argparse
    import os as _os
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default=_os.environ.get('DATA_IMAGES', 'dataset/images'))
    parser.add_argument('--labels', default=_os.environ.get('DATA_LABELS', 'dataset/labels.txt'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    print("Starting ENHANCED CRNN OCR Training")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Data paths (CLI/ENV)
    images_folder = args.images
    labels_file = args.labels

    # Check if data exists
    if not os.path.exists(images_folder):
        print(f"Images folder not found: {images_folder}")
        exit(1)
        
    if not os.path.exists(labels_file):
        print(f"Labels file not found: {labels_file}")
        exit(1)

    # Enhanced image transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.RandomApply([transforms.RandomRotation(degrees=2)], p=0.5),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=2)], p=0.5),
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    print("Loading dataset...")
    try:
        full_dataset = EnhancedOCRDataset(images_folder, labels_file, transform=train_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create validation dataset with different transform
    val_dataset.dataset.transform = val_transform
    
    if len(val_dataset) < 10:
        print(f"Warning: Validation set has only {len(val_dataset)} samples. Consider collecting more data or adjusting train/val split.")
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")

    # Create data loaders
    batch_size = args.batch_size  # Increased batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Enhanced model setup
    num_classes = len(full_dataset.chars)
    model = EnhancedCRNN(num_classes=num_classes).to(device)
    
    # Enhanced loss and optimizer (more stable settings from robust trainer)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Enhanced training parameters
    num_epochs = args.epochs  # configurable epochs
    best_val_loss = float('inf')
    patience = 1000  # Set to a value larger than num_epochs to disable early stopping
    patience_counter = 0
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    print("Starting ENHANCED training...")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training for {num_epochs} epochs with early stopping")
    print("=" * 60)

    start_time = time.time()

    # Checkpoints directory
    ckpt_dir = _os.path.join(_os.getcwd(), 'checkpoints')
    _os.makedirs(ckpt_dir, exist_ok=True)

    # Optional resume
    start_epoch = 0
    if args.resume and _os.path.exists(args.resume):
        try:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt.get('optimizer_state_dict', optimizer.state_dict()))
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            print(f"Resumed from {args.resume} at epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to resume from {args.resume}: {e}")

    def safe_torch_save(obj: dict, path: str):
        tmp = path + '.tmp'
        torch.save(obj, tmp)
        _os.replace(tmp, path)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, full_dataset.chars)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, full_dataset.chars)
        
        # Learning rate scheduling on validation loss (more stable)
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc['char_acc']:.3f} (char), {train_acc['word_acc']:.3f} (word), CER: {train_acc.get('cer', 0):.3f}, WER: {train_acc.get('wer', 0):.3f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc['char_acc']:.3f} (char), {val_acc['word_acc']:.3f} (word), CER: {val_acc.get('cer', 0):.3f}, WER: {val_acc.get('wer', 0):.3f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping (can be effectively disabled by high patience)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            payload = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'chars': full_dataset.chars,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }
            # Save atomically to checkpoints and root
            best_ckpt_path = _os.path.join(ckpt_dir, 'best_model.pth')
            safe_torch_save(payload, best_ckpt_path)
            safe_torch_save(payload, 'enhanced_crnn_model.pth')
            print(f"Best model saved at {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            periodic_path = _os.path.join(ckpt_dir, f'epoch_{epoch+1:04d}.pth')
            safe_torch_save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'chars': full_dataset.chars
            }, periodic_path)

    # Save final model
    safe_torch_save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'chars': full_dataset.chars,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, 'enhanced_crnn_final.pth')
    
    # Save character set
    with open("enhanced_chars.txt", "w", encoding="utf-8") as f:
        for c in full_dataset.chars:
            f.write(c + "\n")
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    print("Training metrics plot saved: training_metrics.png")
    
    # Do not overwrite requirements.txt during training
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("ENHANCED Training complete!")
    print(f"Total training time: {training_time/60:.1f} minutes ({training_time/3600:.1f} hours)")
    print(f"Models saved: enhanced_crnn_model.pth, enhanced_crnn_final.pth")
    print(f"Character set saved: enhanced_chars.txt")
    print("=" * 60)