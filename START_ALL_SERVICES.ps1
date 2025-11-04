# START_ALL_SERVICES.ps1
# Script to start all 3 services for frontend testing

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "LIPIKA - Starting All Services" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check if services are already running
Write-Host "[1/3] Checking Python OCR Service..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "  [OK] Python OCR Service is already running!" -ForegroundColor Green
    }
} catch {
    Write-Host "  [INFO] Starting Python OCR Service..." -ForegroundColor White
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptPath\python-model'; python ocr_service_ar.py" -WindowStyle Minimized
    Start-Sleep -Seconds 3
    Write-Host "  [OK] Python OCR Service started (check minimized window)" -ForegroundColor Green
}

Write-Host "`n[2/3] Checking Java Backend..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/api/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "  [OK] Java Backend is already running!" -ForegroundColor Green
    }
} catch {
    Write-Host "  [INFO] Starting Java Backend..." -ForegroundColor White
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptPath\javabackend'; mvn spring-boot:run" -WindowStyle Minimized
    Start-Sleep -Seconds 5
    Write-Host "  [OK] Java Backend started (check minimized window)" -ForegroundColor Green
    Write-Host "  [NOTE] It may take 10-30 seconds to fully start" -ForegroundColor Gray
}

Write-Host "`n[3/3] Starting React Frontend..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "  [OK] React Frontend is already running!" -ForegroundColor Green
    }
} catch {
    Write-Host "  [INFO] Starting React Frontend..." -ForegroundColor White
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$scriptPath\frontend'; npm run dev" -WindowStyle Minimized
    Start-Sleep -Seconds 3
    Write-Host "  [OK] React Frontend started (check minimized window)" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All Services Starting!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Services:" -ForegroundColor White
Write-Host "  Python OCR:     http://localhost:5000" -ForegroundColor Gray
Write-Host "  Java Backend:   http://localhost:8080" -ForegroundColor Gray
Write-Host "  React Frontend: http://localhost:5173" -ForegroundColor Gray

Write-Host "`nFrontend will open automatically..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Try to open frontend in browser
Start-Process "http://localhost:5173"

Write-Host "`n[OK] Frontend opened in browser!" -ForegroundColor Green
Write-Host "`nWait for all services to fully start before testing." -ForegroundColor Yellow
Write-Host "Check minimized PowerShell windows for service status.`n" -ForegroundColor Gray
