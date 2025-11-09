# PowerShell script to start all Lipika OCR services
# Usage: .\START_ALL_SERVICES.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Lipika OCR System - Startup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Step 1: Test Model
Write-Host "[1/4] Testing trained model..." -ForegroundColor Yellow
Set-Location "python-model"
$testResult = python test_trained_model.py 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Model test failed!" -ForegroundColor Red
    Write-Host $testResult
    exit 1
}
Write-Host "[OK] Model test passed" -ForegroundColor Green
Write-Host ""

# Step 2: Start Python OCR Service
Write-Host "[2/4] Starting Python OCR Service (Port 5000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir\python-model'; python ocr_service_ar.py" -WindowStyle Normal
Write-Host "[OK] Python service starting in new window" -ForegroundColor Green
Write-Host "   Waiting 5 seconds for service to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 5
Write-Host ""

# Step 3: Start Java Backend
Write-Host "[3/4] Starting Java Backend (Port 8080)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptDir\javabackend'; mvn spring-boot:run" -WindowStyle Normal
Write-Host "[OK] Java backend starting in new window" -ForegroundColor Green
Write-Host "   Waiting 10 seconds for service to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 10
Write-Host ""

# Step 4: Start React Frontend
Write-Host "[4/4] Starting React Frontend (Port 3000)..." -ForegroundColor Yellow
Set-Location "$scriptDir\frontend"

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "   Installing dependencies..." -ForegroundColor Gray
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] npm install failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[OK] Starting frontend..." -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All services are starting!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  - Python OCR Service: http://localhost:5000" -ForegroundColor White
Write-Host "  - Java Backend:       http://localhost:8080" -ForegroundColor White
Write-Host "  - React Frontend:     http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the frontend server" -ForegroundColor Gray
Write-Host ""

# Start the frontend (this will block)
npm run dev
