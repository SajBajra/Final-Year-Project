Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Testing Lipika Services" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Test Python OCR Service
Write-Host "[1/2] Testing Python OCR Service (port 5000)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -Method Get -TimeoutSec 5 -UseBasicParsing
    Write-Host "  Status Code: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Response: $($response.Content)" -ForegroundColor Green
    Write-Host "  [OK] Python OCR Service is responding!" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python OCR Service is not responding: $_" -ForegroundColor Red
    Write-Host "  → Make sure to run: cd python-model && python ocr_service_ar.py" -ForegroundColor Yellow
}

Write-Host ""

# Test Java Backend
Write-Host "[2/2] Testing Java Backend (port 8080)..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/api/health" -Method Get -TimeoutSec 5 -UseBasicParsing
    Write-Host "  Status Code: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Response: $($response.Content)" -ForegroundColor Green
    Write-Host "  [OK] Java Backend is responding!" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Java Backend is not responding: $_" -ForegroundColor Red
    Write-Host "  → Make sure to run Java Backend from Eclipse or: cd javabackend && mvn spring-boot:run" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Test Complete" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
