# LEGEAI - Automated Setup Script for Windows
# This script automates the installation and setup process

Write-Host "🚀 LEGEAI - Automated Setup Script" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Function to print colored output
function Print-Success {
    param($Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Print-Info {
    param($Message)
    Write-Host "ℹ $Message" -ForegroundColor Blue
}

function Print-Warning {
    param($Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Print-Error {
    param($Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Check if Python is installed
Print-Info "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Print-Success "Python found: $pythonVersion"
} catch {
    Print-Error "Python not found. Please install Python 3.9 or higher."
    exit 1
}

# Check if Node.js is installed
Print-Info "Checking Node.js installation..."
try {
    $nodeVersion = node --version
    Print-Success "Node.js found: $nodeVersion"
} catch {
    Print-Error "Node.js not found. Please install Node.js 18.17 or higher."
    exit 1
}

# Check if npm is installed
Print-Info "Checking npm installation..."
try {
    $npmVersion = npm --version
    Print-Success "npm found: v$npmVersion"
} catch {
    Print-Error "npm not found. Please install npm."
    exit 1
}

Write-Host ""
Print-Info "Step 1: Creating Python virtual environment..."
if (-not (Test-Path "venv")) {
    python -m venv venv
    Print-Success "Virtual environment created"
} else {
    Print-Warning "Virtual environment already exists"
}

Write-Host ""
Print-Info "Step 2: Activating virtual environment..."
& .\venv\Scripts\Activate.ps1
Print-Success "Virtual environment activated"

Write-Host ""
Print-Info "Step 3: Upgrading pip..."
python -m pip install --upgrade pip --quiet
Print-Success "pip upgraded"

Write-Host ""
Print-Info "Step 4: Installing Python dependencies..."
Print-Warning "This may take 5-10 minutes..."
pip install -r requirements.txt --quiet
Print-Success "Python dependencies installed"

Write-Host ""
Print-Info "Step 5: Creating .env file..."
if (-not (Test-Path ".env")) {
    @"
# Database Configuration
USE_DB=false
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/legeai

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# JWT Secret
JWT_SECRET_KEY=dev-secret-key-change-in-production

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# CORS Origins
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Print-Success ".env file created"
} else {
    Print-Warning ".env file already exists"
}

Write-Host ""
Print-Info "Step 6: Installing frontend dependencies..."
Set-Location frontend
if (-not (Test-Path "node_modules")) {
    Print-Warning "This may take 3-5 minutes..."
    npm install --silent
    Print-Success "Frontend dependencies installed"
} else {
    Print-Warning "node_modules already exists, running npm install to update..."
    npm install --silent
    Print-Success "Frontend dependencies updated"
}

Write-Host ""
Print-Info "Step 7: Creating frontend .env.local file..."
if (-not (Test-Path ".env.local")) {
    @"
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Environment
NODE_ENV=development
"@ | Out-File -FilePath ".env.local" -Encoding UTF8
    Print-Success "Frontend .env.local file created"
} else {
    Print-Warning "Frontend .env.local file already exists"
}

Set-Location ..

Write-Host ""
Write-Host "==================================" -ForegroundColor Green
Print-Success "Setup completed successfully!"
Write-Host "==================================" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Start the backend (Terminal 1):" -ForegroundColor White
Write-Host "   .\run_backend.ps1" -ForegroundColor Blue
Write-Host ""
Write-Host "2. Start the frontend (Terminal 2):" -ForegroundColor White
Write-Host "   cd frontend" -ForegroundColor Blue
Write-Host "   npm run dev" -ForegroundColor Blue
Write-Host ""
Write-Host "3. Open your browser:" -ForegroundColor White
Write-Host "   Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host "   API Docs: http://localhost:8000/v1/docs" -ForegroundColor Green
Write-Host ""
Write-Host "4. Login with test credentials:" -ForegroundColor White
Write-Host "   Email: admin@legeai.dev" -ForegroundColor Yellow
Write-Host "   Password: AdminPass123!" -ForegroundColor Yellow
Write-Host ""
Print-Info "For more information, see SETUP_GUIDE.md"
Write-Host ""

