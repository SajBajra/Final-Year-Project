# Lipika - Complete Service Startup Script
# This script verifies prerequisites and starts both OCR service and frontend

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lipika - Complete System Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  [OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Python not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Yellow
    exit 1
}

# Step 2: Check Python Dependencies
Write-Host "`n[2/6] Checking Python dependencies..." -ForegroundColor Yellow
Set-Location python-model
try {
    python -c "import flask, flask_cors, torch, cv2, PIL; print('OK')" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] All Python dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Some dependencies missing, installing..." -ForegroundColor Yellow
        pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] Dependencies installed" -ForegroundColor Green
        } else {
            Write-Host "  [FAIL] Failed to install dependencies" -ForegroundColor Red
            Set-Location ..
            exit 1
        }
    }
} catch {
    Write-Host "  [FAIL] Error checking dependencies" -ForegroundColor Red
    Set-Location ..
    exit 1
}
Set-Location ..

# Step 3: Check Model File
Write-Host "`n[3/6] Checking model file..." -ForegroundColor Yellow
if (Test-Path "python-model\best_character_crnn.pth") {
    $modelSize = (Get-Item "python-model\best_character_crnn.pth").Length / 1MB
    Write-Host "  [OK] Model file found ($([math]::Round($modelSize, 2)) MB)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Model file not found!" -ForegroundColor Yellow
    Write-Host "  The service will start but OCR won't work without the model." -ForegroundColor Yellow
    Write-Host "  Train the model first: python python-model\train_character_crnn.py --epochs 100" -ForegroundColor Cyan
}

# Step 4: Check Node.js
Write-Host "`n[4/6] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  [OK] Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Node.js not found!" -ForegroundColor Red
    Write-Host "  Please install Node.js 18+ from https://nodejs.org/" -ForegroundColor Yellow
    Write-Host "  Frontend will not be started." -ForegroundColor Yellow
    $nodeAvailable = $false
}

# Step 5: Check Frontend Dependencies
if ($nodeAvailable -ne $false) {
    Write-Host "`n[5/6] Checking frontend dependencies..." -ForegroundColor Yellow
    if (Test-Path "frontend\node_modules") {
        Write-Host "  [OK] Frontend dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Frontend dependencies not found, installing..." -ForegroundColor Yellow
        Set-Location frontend
        npm install
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] Frontend dependencies installed" -ForegroundColor Green
        } else {
            Write-Host "  [WARN] Frontend dependencies installation had issues" -ForegroundColor Yellow
        }
        Set-Location ..
    }
} else {
    Write-Host "`n[5/6] Skipping frontend (Node.js not available)..." -ForegroundColor Yellow
}

# Step 6: Start Services
Write-Host "`n[6/6] Starting services..." -ForegroundColor Yellow
Write-Host ""

# Start OCR Service
Write-Host "Starting OCR Service..." -ForegroundColor Cyan
Write-Host "  → http://localhost:5000" -ForegroundColor Gray
Set-Location python-model
$ocrProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Lipika OCR Service' -ForegroundColor Green; Write-Host ''; python ocr_service_ar.py" -PassThru
Set-Location ..

# Wait for OCR service to initialize
Start-Sleep -Seconds 3

# Start Frontend (if Node.js available)
if ($nodeAvailable -ne $false) {
    Write-Host "Starting Frontend..." -ForegroundColor Cyan
    Write-Host "  → http://localhost:5173" -ForegroundColor Gray
    Set-Location frontend
    $frontendProcess = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Lipika Frontend' -ForegroundColor Green; Write-Host ''; npm run dev" -PassThru
    Set-Location ..
} else {
    Write-Host "Frontend not started (Node.js not available)" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "OCR Service:    http://localhost:5000" -ForegroundColor Green
if ($nodeAvailable -ne $false) {
    Write-Host "Frontend:       http://localhost:5173" -ForegroundColor Green
}
Write-Host ""
Write-Host "Test endpoints:" -ForegroundColor Yellow
Write-Host "  • Health:      http://localhost:5000/health" -ForegroundColor Gray
Write-Host "  • API Info:    http://localhost:5000/" -ForegroundColor Gray
if ($nodeAvailable -ne $false) {
    Write-Host "  • Frontend:    http://localhost:5173" -ForegroundColor Gray
}
Write-Host ""
Write-Host "Services are running in separate windows." -ForegroundColor Cyan
Write-Host "Close those windows to stop the services." -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit this script (services will continue running)..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
