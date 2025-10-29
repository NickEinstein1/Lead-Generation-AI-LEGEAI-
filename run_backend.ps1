# Script to run the LEAGAI backend from the new backend folder structure

Write-Host "🚀 Starting LEAGAI Backend..." -ForegroundColor Green
Write-Host "Backend location: backend/api/main.py" -ForegroundColor Cyan
Write-Host "Running from root directory to ensure proper module resolution" -ForegroundColor Cyan

# Set environment variables
$env:PYTHONPATH = "."
$env:USE_DB = "false"

# Run the backend from ROOT directory (not from backend folder)
# This ensures Python can find the 'backend' module
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

