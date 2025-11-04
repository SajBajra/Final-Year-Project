# Lipika Setup Script
# Ensures all dependencies are properly installed

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lipika - Dependency Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Python
Write-Host "`n[1/4] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
Write-Host "`n[2/4] Checking virtual environment..." -ForegroundColor Yellow
$venvPath = "python-model\venv"
if (Test-Path $venvPath) {
    Write-Host "✓ Virtual environment exists" -ForegroundColor Green
    Write-Host "  Activating virtual environment..." -ForegroundColor Cyan
    & "$venvPath\Scripts\Activate.ps1"
} else {
    Write-Host "⚠ Virtual environment not found" -ForegroundColor Yellow
    $create = Read-Host "Create virtual environment? (recommended) [Y/n]"
    if ($create -ne "n" -and $create -ne "N") {
        Write-Host "  Creating virtual environment..." -ForegroundColor Cyan
        cd python-model
        python -m venv venv
        cd ..
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
        Write-Host "  Activating..." -ForegroundColor Cyan
        & "$venvPath\Scripts\Activate.ps1"
    } else {
        Write-Host "  Continuing without virtual environment..." -ForegroundColor Yellow
    }
}

# Install Python dependencies
Write-Host "`n[3/4] Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "  Installing from requirements.txt..." -ForegroundColor Cyan
cd python-model
pip install --upgrade pip
pip install -r requirements.txt
cd ..

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All Python dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install some dependencies" -ForegroundColor Red
    Write-Host "  Please check the error messages above" -ForegroundColor Yellow
    exit 1
}

# Check Node.js for frontend
Write-Host "`n[4/4] Checking Node.js (for frontend)..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js $nodeVersion" -ForegroundColor Green
    
    # Install frontend dependencies
    Write-Host "`n  Installing frontend dependencies..." -ForegroundColor Cyan
    cd frontend
    if (Test-Path "package.json") {
        npm install
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Frontend dependencies installed!" -ForegroundColor Green
        } else {
            Write-Host "⚠ Frontend dependencies installation had issues" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠ package.json not found in frontend/" -ForegroundColor Yellow
    }
    cd ..
} catch {
    Write-Host "⚠ Node.js not found (optional for now)" -ForegroundColor Yellow
    Write-Host "  Install from https://nodejs.org/ to use frontend" -ForegroundColor Cyan
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n✓ All dependencies are now installed" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Train the model (if not already done):" -ForegroundColor Cyan
Write-Host "     cd python-model" -ForegroundColor White
Write-Host "     python train_character_crnn.py --epochs 100" -ForegroundColor White
Write-Host "`n  2. Start the OCR service:" -ForegroundColor Cyan
Write-Host "     cd python-model" -ForegroundColor White
Write-Host "     python ocr_service_ar.py" -ForegroundColor White
Write-Host "`n  3. Start the frontend (in another terminal):" -ForegroundColor Cyan
Write-Host "     cd frontend" -ForegroundColor White
Write-Host "     npm run dev" -ForegroundColor White
Write-Host "`nOr use: .\start_services.ps1" -ForegroundColor Cyan
