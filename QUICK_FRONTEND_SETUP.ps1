# Quick Frontend Setup Script for Lipika
# Checks for Node.js and sets up frontend dependencies

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lipika - Frontend Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for Node.js
Write-Host "[1/3] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  [OK] Node.js $nodeVersion found" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] Node.js not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Node.js first:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://nodejs.org/" -ForegroundColor Cyan
    Write-Host "  2. Download LTS version" -ForegroundColor Cyan
    Write-Host "  3. Install it" -ForegroundColor Cyan
    Write-Host "  4. Close and reopen this terminal" -ForegroundColor Cyan
    Write-Host "  5. Run this script again" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or see: NODEJS_INSTALLATION_GUIDE.md" -ForegroundColor Yellow
    exit 1
}

# Check npm
Write-Host "`n[2/3] Checking npm..." -ForegroundColor Yellow
try {
    $npmVersion = npm --version 2>&1
    Write-Host "  [OK] npm $npmVersion found" -ForegroundColor Green
} catch {
    Write-Host "  [FAIL] npm not found!" -ForegroundColor Red
    Write-Host "  npm should come with Node.js. Please reinstall Node.js." -ForegroundColor Yellow
    exit 1
}

# Check if dependencies are already installed
Write-Host "`n[3/3] Checking frontend dependencies..." -ForegroundColor Yellow
if (Test-Path "frontend\node_modules") {
    Write-Host "  [OK] Dependencies already installed" -ForegroundColor Green
    Write-Host ""
    Write-Host "Dependencies are ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start the frontend:" -ForegroundColor Yellow
    Write-Host "  cd frontend" -ForegroundColor Cyan
    Write-Host "  npm run dev" -ForegroundColor Cyan
    Write-Host ""
    $startNow = Read-Host "Start frontend now? (Y/n)"
    if ($startNow -ne "n" -and $startNow -ne "N") {
        Write-Host ""
        Write-Host "Starting frontend..." -ForegroundColor Cyan
        Set-Location frontend
        npm run dev
    }
} else {
    Write-Host "  [WARN] Dependencies not found" -ForegroundColor Yellow
    Write-Host "  Installing dependencies..." -ForegroundColor Cyan
    Write-Host ""
    
    Set-Location frontend
    npm install
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "  [OK] Dependencies installed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Frontend setup complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "To start the frontend:" -ForegroundColor Yellow
        Write-Host "  npm run dev" -ForegroundColor Cyan
        Write-Host ""
        $startNow = Read-Host "Start frontend now? (Y/n)"
        if ($startNow -ne "n" -and $startNow -ne "N") {
            Write-Host ""
            Write-Host "Starting frontend..." -ForegroundColor Cyan
            npm run dev
        }
    } else {
        Write-Host ""
        Write-Host "  [FAIL] Failed to install dependencies" -ForegroundColor Red
        Write-Host "  Please check the error messages above" -ForegroundColor Yellow
        Set-Location ..
        exit 1
    }
}

Set-Location ..
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend will run on: http://localhost:5173" -ForegroundColor Green
Write-Host ""
Write-Host "Make sure OCR service is running:" -ForegroundColor Yellow
Write-Host "  cd python-model" -ForegroundColor Cyan
Write-Host "  python ocr_service_ar.py" -ForegroundColor Cyan
Write-Host ""
