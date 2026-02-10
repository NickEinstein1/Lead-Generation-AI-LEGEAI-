# Script to run the LEAGAI backend from the new backend folder structure

Write-Host "🚀 Starting LEAGAI Backend..." -ForegroundColor Green
Write-Host "Backend location: backend/api/main.py" -ForegroundColor Cyan
Write-Host "Running from root directory to ensure proper module resolution" -ForegroundColor Cyan

# Set environment variables
$env:PYTHONPATH = "."

# Prefer DB-backed mode by default. If USE_DB is not already set in the
# environment (e.g., via .env), default it to "true" so that the backend
# uses the real database instead of in-memory demo mode.
if (-not $env:USE_DB) {
    $env:USE_DB = "true"
}

# Activate virtual environment if it exists
$venvPython = $null
if (Test-Path "venv\Scripts\python.exe") {
    $venvPython = "venv\Scripts\python.exe"
} elseif (Test-Path ".venv\Scripts\python.exe") {
    $venvPython = ".venv\Scripts\python.exe"
} else {
    $venvPython = "python"
}

Write-Host "Using Python: $venvPython" -ForegroundColor Cyan

# Run the backend from ROOT directory (not from backend folder)
# This ensures Python can find the 'backend' module
& $venvPython -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

