# Lipika - Complete Model Training
# This script starts extended training with maximum epochs

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lipika - Complete Model Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

# Check for existing checkpoint
$checkpointPath = "best_character_crnn_improved.pth"
$resumeTraining = $false

if (Test-Path $checkpointPath) {
    Write-Host "[INFO] Found existing checkpoint: $checkpointPath" -ForegroundColor Yellow
    
    try {
        $checkpointInfo = python -c "import torch; ckpt = torch.load('best_character_crnn_improved.pth', map_location='cpu'); print(f\"Epoch: {ckpt.get('epoch', 0)}\"); print(f\"Val Acc: {ckpt.get('val_acc', 0):.2f}%\")" 2>&1
        Write-Host $checkpointInfo
        
        Write-Host "`nOptions:" -ForegroundColor Cyan
        Write-Host "  1. Resume from checkpoint (continue training)" -ForegroundColor White
        Write-Host "  2. Start fresh (overwrite existing model)" -ForegroundColor White
        $choice = Read-Host "`nChoose option (1 or 2, default: 1)"
        
        if ([string]::IsNullOrWhiteSpace($choice) -or $choice -eq "1") {
            $resumeTraining = $true
            Write-Host "[OK] Will resume from checkpoint" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Will start fresh training" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[WARN] Could not read checkpoint info" -ForegroundColor Yellow
        $resumeTraining = $true
    }
} else {
    Write-Host "[INFO] No checkpoint found. Starting fresh training." -ForegroundColor Yellow
}

# Training parameters
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Training Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$epochs = Read-Host "Enter number of epochs (default: 300, recommended: 300-500)"
if ([string]::IsNullOrWhiteSpace($epochs)) {
    $epochs = 300
} else {
    $epochs = [int]$epochs
}

$batchSize = Read-Host "Enter batch size (default: 64, reduce if out of memory)"
if ([string]::IsNullOrWhiteSpace($batchSize)) {
    $batchSize = 64
} else {
    $batchSize = [int]$batchSize
}

$learningRate = Read-Host "Enter learning rate (default: 0.001, use 0.0001 for fine-tuning)"
if ([string]::IsNullOrWhiteSpace($learningRate)) {
    $learningRate = 0.001
} else {
    $learningRate = [double]$learningRate
}

# Dataset paths
$imagesPath = "../prepared_dataset/images"
$trainLabelsPath = "../prepared_dataset/train_labels.txt"
$valLabelsPath = "../prepared_dataset/val_labels.txt"

# Verify dataset exists
if (-not (Test-Path $imagesPath)) {
    Write-Host "[ERROR] Dataset not found at: $imagesPath" -ForegroundColor Red
    Write-Host "Please make sure the dataset is prepared first." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Final Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Images: $imagesPath"
Write-Host "Train labels: $trainLabelsPath"
Write-Host "Val labels: $valLabelsPath"
Write-Host "Epochs: $epochs"
Write-Host "Batch size: $batchSize"
Write-Host "Learning rate: $learningRate"
if ($resumeTraining) {
    Write-Host "Resume: Yes (from $checkpointPath)" -ForegroundColor Green
} else {
    Write-Host "Resume: No (fresh training)" -ForegroundColor Yellow
}
Write-Host ""

$confirm = Read-Host "Start training? (Y/n)"
if ($confirm -ne "" -and $confirm.ToLower() -ne "y") {
    Write-Host "Training cancelled." -ForegroundColor Yellow
    exit 0
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Starting Training..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will take a while depending on:" -ForegroundColor Yellow
Write-Host "  • Number of epochs: $epochs" -ForegroundColor Gray
Write-Host "  • Your hardware (CPU/GPU)" -ForegroundColor Gray
Write-Host "  • Batch size: $batchSize" -ForegroundColor Gray
Write-Host ""
Write-Host "Estimated time:" -ForegroundColor Yellow
$estimatedHours = [math]::Round($epochs * 0.02, 1)  # Rough estimate
Write-Host "  CPU: ~$estimatedHours hours" -ForegroundColor Gray
Write-Host "  GPU: ~$([math]::Round($epochs * 0.005, 1)) hours" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop training (model will be saved)" -ForegroundColor Cyan
Write-Host ""

# Build command
$cmd = "python train_character_crnn_improved.py --images $imagesPath --train_labels $trainLabelsPath --val_labels $valLabelsPath --epochs $epochs --batch_size $batchSize --lr $learningRate"

if ($resumeTraining) {
    $cmd += " --resume $checkpointPath"
}

Write-Host "Command: $cmd" -ForegroundColor Gray
Write-Host ""
Write-Host "Starting in 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Execute training
Invoke-Expression $cmd

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Model saved as: best_character_crnn_improved.pth" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Check the training curves: training_curves_improved.png" -ForegroundColor Gray
Write-Host "  2. Start OCR service: python ocr_service_ar.py" -ForegroundColor Gray
Write-Host "  3. Test with images!" -ForegroundColor Gray
