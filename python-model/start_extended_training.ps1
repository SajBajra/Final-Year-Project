# PowerShell script to start extended training
# Automatically detects if checkpoint exists and resumes if available

Write-Host "=== Lipika Extended Training ===" -ForegroundColor Cyan
Write-Host ""

# Navigate to python-model directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check for existing checkpoint
$checkpointPath = "best_character_crnn_improved.pth"
$resumeFlag = ""

if (Test-Path $checkpointPath) {
    Write-Host "[OK] Found existing checkpoint: $checkpointPath" -ForegroundColor Green
    
    # Try to read checkpoint info
    Write-Host "`nChecking checkpoint information..." -ForegroundColor Yellow
    try {
        $pythonCheck = @"
import torch
import sys
checkpoint = torch.load('best_character_crnn_improved.pth', map_location='cpu')
epoch = checkpoint.get('epoch', 'N/A')
val_acc = checkpoint.get('val_acc', 'N/A')
print(f"Last epoch: {epoch}")
print(f"Best validation accuracy: {val_acc:.2f}%")
"@
        $result = python -c $pythonCheck 2>&1
        Write-Host $result
        
        Write-Host "`n[INFO] You can resume from this checkpoint or train fresh." -ForegroundColor Cyan
        $choice = Read-Host "Resume from checkpoint? (Y/n)"
        
        if ($choice -eq "" -or $choice.ToLower() -eq "y") {
            $resumeFlag = "--resume $checkpointPath"
            Write-Host "[OK] Will resume from checkpoint" -ForegroundColor Green
        } else {
            Write-Host "[INFO] Will start fresh training (existing checkpoint will be overwritten)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[WARN] Could not read checkpoint info, will start fresh" -ForegroundColor Yellow
    }
} else {
    Write-Host "[INFO] No existing checkpoint found. Starting fresh training." -ForegroundColor Yellow
}

# Training parameters
$epochs = Read-Host "`nEnter number of epochs (default: 250)"
if ([string]::IsNullOrWhiteSpace($epochs)) {
    $epochs = 250
}

$batchSize = Read-Host "Enter batch size (default: 64)"
if ([string]::IsNullOrWhiteSpace($batchSize)) {
    $batchSize = 64
}

$learningRate = Read-Host "Enter learning rate (default: 0.001, use 0.0001 for fine-tuning)"
if ([string]::IsNullOrWhiteSpace($learningRate)) {
    $learningRate = 0.001
}

# Dataset paths
$imagesPath = "../prepared_dataset/images"
$trainLabelsPath = "../prepared_dataset/train_labels.txt"
$valLabelsPath = "../prepared_dataset/val_labels.txt"

# Check if dataset exists
if (-not (Test-Path $imagesPath)) {
    Write-Host "[ERROR] Dataset not found at: $imagesPath" -ForegroundColor Red
    Write-Host "Please make sure you have prepared the dataset first." -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Training Configuration ===" -ForegroundColor Cyan
Write-Host "Images: $imagesPath"
Write-Host "Train labels: $trainLabelsPath"
Write-Host "Val labels: $valLabelsPath"
Write-Host "Epochs: $epochs"
Write-Host "Batch size: $batchSize"
Write-Host "Learning rate: $learningRate"
if ($resumeFlag) {
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

Write-Host "`nStarting training..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop training" -ForegroundColor Gray
Write-Host ""

# Build command
$cmd = "python train_character_crnn_improved.py --images $imagesPath --train_labels $trainLabelsPath --val_labels $valLabelsPath --epochs $epochs --batch_size $batchSize --lr $learningRate"

if ($resumeFlag) {
    $cmd += " $resumeFlag"
}

# Execute training
Invoke-Expression $cmd
