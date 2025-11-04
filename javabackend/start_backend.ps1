# PowerShell script to start Java Backend
# Usage: .\start_backend.ps1

Write-Host "=== Starting Lipika Java Backend ===" -ForegroundColor Cyan
Write-Host ""

# Check if Maven is installed
Write-Host "Checking Maven..." -ForegroundColor Yellow
try {
    $mvnVersion = mvn --version 2>&1
    Write-Host "[OK] Maven is installed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Maven is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Maven first:" -ForegroundColor Yellow
    Write-Host "  winget install Apache.Maven" -ForegroundColor White
    exit 1
}

# Check if Java is installed
Write-Host "Checking Java..." -ForegroundColor Yellow
try {
    $javaVersion = java -version 2>&1
    Write-Host "[OK] Java is installed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Java is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Java 17 or higher" -ForegroundColor Yellow
    exit 1
}

# Navigate to javabackend directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host ""
Write-Host "Building project..." -ForegroundColor Yellow
mvn clean install -DskipTests

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[OK] Build successful!" -ForegroundColor Green
Write-Host ""
Write-Host "Starting Spring Boot application..." -ForegroundColor Yellow
Write-Host "The server will start on http://localhost:8080" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Start the Spring Boot application
mvn spring-boot:run
