@echo off
REM Start Backend Server for LEGEAI (Windows)
REM This script activates the virtual environment and starts the FastAPI backend

echo.
echo ========================================
echo   Starting LEGEAI Backend Server
echo ========================================
echo.

REM Activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
) else (
    echo ✗ Virtual environment not found at .venv\Scripts\activate.bat
    echo   Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo   Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo ✓ Starting FastAPI server on http://127.0.0.1:8000
echo.
echo   API Documentation: http://127.0.0.1:8000/v1/docs
echo   Health Check:      http://127.0.0.1:8000/health
echo.
echo Press CTRL+C to stop the server
echo.

REM Start the backend server
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

