# Start Backend Server for LEGEAI (Windows PowerShell)
# This script activates the virtual environment and starts the FastAPI backend

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting LEGEAI Backend Server" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "✗ Virtual environment not found" -ForegroundColor Red
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\Scripts\Activate.ps1
    Write-Host "  Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "✓ Starting FastAPI server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host ""
Write-Host "  API Documentation: http://127.0.0.1:8000/v1/docs" -ForegroundColor Cyan
Write-Host "  Health Check:      http://127.0.0.1:8000/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the backend server
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

