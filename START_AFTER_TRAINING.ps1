# Start All Services After Training - Lipika
# This script starts the Python OCR service, Java backend, and Frontend

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Lipika Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if model exists
Write-Host "[1/4] Checking trained model..." -ForegroundColor Yellow
if (Test-Path "python-model/best_character_crnn_improved.pth") {
    $model = Get-Item "python-model/best_character_crnn_improved.pth"
    $sizeMB = [math]::Round($model.Length / 1MB, 2)
    Write-Host "  [OK] Model found: $sizeMB MB" -ForegroundColor Green
} else {
    Write-Host "  [WARN] Model not found! Make sure training completed." -ForegroundColor Yellow
}

# Start Python OCR Service
Write-Host "`n[2/4] Starting Python OCR Service..." -ForegroundColor Yellow
Write-Host "  → http://localhost:5000" -ForegroundColor Cyan
Set-Location python-model
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Lipika OCR Service' -ForegroundColor Green; Write-Host ''; python ocr_service_ar.py"
Set-Location ..

# Wait for OCR service to start
Start-Sleep -Seconds 3

# Start Java Backend
Write-Host "`n[3/4] Starting Java Backend..." -ForegroundColor Yellow
Write-Host "  → http://localhost:8080" -ForegroundColor Cyan
Write-Host "  Note: Start manually from Eclipse or run: cd javabackend; mvn spring-boot:run" -ForegroundColor Gray
Set-Location javabackend
if (Test-Path "pom.xml") {
    Write-Host "  [INFO] You can start manually with: mvn spring-boot:run" -ForegroundColor Gray
} else {
    Write-Host "  [WARN] Java backend not found!" -ForegroundColor Yellow
}
Set-Location ..

# Start Frontend
Write-Host "`n[4/4] Starting Frontend..." -ForegroundColor Yellow
Write-Host "  → http://localhost:5173" -ForegroundColor Cyan
Set-Location frontend
if (Test-Path "package.json") {
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host 'Lipika Frontend' -ForegroundColor Green; Write-Host ''; npm run dev"
} else {
    Write-Host "  [WARN] Frontend not found!" -ForegroundColor Yellow
}
Set-Location ..

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  • Python OCR:    http://localhost:5000" -ForegroundColor Green
Write-Host "  • Java Backend:  http://localhost:8080" -ForegroundColor Green
Write-Host "  • Frontend:      http://localhost:5173" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Start Java Backend from Eclipse OR:" -ForegroundColor Gray
Write-Host "     cd javabackend" -ForegroundColor Gray
Write-Host "     mvn spring-boot:run" -ForegroundColor Gray
Write-Host "  2. Open Frontend: http://localhost:5173" -ForegroundColor Gray
Write-Host "  3. Upload a Ranjana image to test OCR!" -ForegroundColor Gray
Write-Host ""
Write-Host "Services are running in separate windows." -ForegroundColor Cyan
Write-Host "Close those windows to stop the services." -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit this script (services will continue running)..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
