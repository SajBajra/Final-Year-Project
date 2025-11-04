# Lipika Service Startup Script
# This script starts both the OCR service and frontend

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Lipika Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if Python is available
Write-Host "`n[1/3] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check if Node.js is available
Write-Host "`n[2/3] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Node.js not found! Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Start OCR Service
Write-Host "`n[3/3] Starting OCR Service..." -ForegroundColor Yellow
Write-Host "  → http://localhost:5000" -ForegroundColor Cyan
Set-Location python-model
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python ocr_service_ar.py"
Set-Location ..

# Wait a bit for OCR service to start
Start-Sleep -Seconds 3

# Start Frontend
Write-Host "`n[4/4] Starting Frontend..." -ForegroundColor Yellow
Write-Host "  → http://localhost:3000" -ForegroundColor Cyan
Set-Location frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev"
Set-Location ..

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nOCR Service: http://localhost:5000" -ForegroundColor Green
Write-Host "Frontend:     http://localhost:3000" -ForegroundColor Green
Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
