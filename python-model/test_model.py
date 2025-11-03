"""
Quick test script to verify the trained model loads correctly
"""

import torch
import sys

def test_model():
    try:
        print("Testing model loading...")
        print("=" * 60)
        
        # Load checkpoint
        checkpoint_path = "best_character_crnn.pth"
        print(f"Loading: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract info
        chars = checkpoint.get('chars', [])
        num_classes = len(chars)
        val_acc = checkpoint.get('val_acc', 0)
        epoch = checkpoint.get('epoch', 0)
        
        print(f"\n✓ Model loaded successfully!")
        print(f"  - Classes: {num_classes}")
        print(f"  - Characters: {len([c for c in chars if c])} non-blank")
        print(f"  - Validation Accuracy: {val_acc:.2f}%")
        print(f"  - Epoch: {epoch}")
        
        if chars:
            print(f"\n  - Character set: {''.join([c for c in chars[:20] if c])}...")
        
        # Test forward pass - Import model directly
        import torch.nn as nn
        
        # Define model architecture inline
        class CharacterCRNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 32, 3, 1, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, 3, 1, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.Conv2d(256, 256, 3, 1, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, 1, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.rnn = nn.LSTM(512, 256, 2, bidirectional=True, batch_first=True, dropout=0.3)
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
                conv_features = self.cnn(x)
                b, c, h, w = conv_features.size()
                conv_features = conv_features.view(b, c, -1).permute(0, 2, 1)
                rnn_out, _ = self.rnn(conv_features)
                return self.classifier(rnn_out.squeeze(1))
        
        model = CharacterCRNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, 64, 64)
        output = model(dummy_input)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output classes: {output.shape[1]}")
        
        print("\n" + "=" * 60)
        print("✅ MODEL TEST PASSED - Ready for OCR!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)

