# PowerShell script to retrain model with Ranjana characters
# This will train a new model that predicts Ranjana Unicode characters

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "LIPIKA - Retrain Model with Ranjana Characters" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "train_character_crnn_improved.py")) {
    Write-Host "[ERROR] Please run this script from the python-model directory" -ForegroundColor Red
    exit 1
}

# Check if Ranjana labels exist
$trainLabels = "../prepared_dataset/train_labels_ranjana.txt"
$valLabels = "../prepared_dataset/val_labels_ranjana.txt"
$imagesFolder = "../prepared_dataset/images"

if (-not (Test-Path $trainLabels)) {
    Write-Host "[ERROR] Ranjana training labels not found: $trainLabels" -ForegroundColor Red
    Write-Host "[INFO] Run convert_labels_to_ranjana.py first to create Ranjana labels" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $valLabels)) {
    Write-Host "[ERROR] Ranjana validation labels not found: $valLabels" -ForegroundColor Red
    Write-Host "[INFO] Run convert_labels_to_ranjana.py first to create Ranjana labels" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $imagesFolder)) {
    Write-Host "[ERROR] Images folder not found: $imagesFolder" -ForegroundColor Red
    exit 1
}

# Backup existing model
$modelPath = "best_character_crnn_improved.pth"
if (Test-Path $modelPath) {
    $backupPath = "${modelPath}.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item $modelPath $backupPath
    Write-Host "[INFO] Backed up existing model to: $backupPath" -ForegroundColor Green
}

Write-Host ""
Write-Host "[INFO] Starting training with Ranjana labels..." -ForegroundColor Yellow
Write-Host "[INFO] This may take a while (150 epochs by default)" -ForegroundColor Yellow
Write-Host ""

# Run training
python retrain_with_ranjana.py `
    --images $imagesFolder `
    --train_labels $trainLabels `
    --val_labels $valLabels `
    --epochs 150 `
    --batch_size 64 `
    --lr 0.001

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "[SUCCESS] Training completed!" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "[INFO] Model saved as: best_character_crnn_improved.pth" -ForegroundColor Cyan
    Write-Host "[INFO] Restart the OCR service to use the new model" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "[ERROR] Training failed. Check the error messages above." -ForegroundColor Red
    exit 1
}

