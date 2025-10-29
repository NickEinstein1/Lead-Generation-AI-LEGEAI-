# LEAGAI Development Environment Startup Script
# This script starts the backend, creates test accounts, and starts the frontend

Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          🚀 LEAGAI Development Environment Startup            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    $connection = New-Object System.Net.Sockets.TcpClient
    try {
        $connection.Connect("127.0.0.1", $Port)
        $connection.Close()
        return $true
    }
    catch {
        return $false
    }
}

# Function to start a process in a new window
function Start-ProcessInWindow {
    param(
        [string]$Title,
        [string]$Command,
        [string]$WorkingDirectory
    )
    
    Write-Host "📌 Starting: $Title" -ForegroundColor Yellow
    Write-Host "   Command: $Command" -ForegroundColor Gray
    Write-Host "   Location: $WorkingDirectory" -ForegroundColor Gray
    
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = "powershell.exe"
    $processInfo.Arguments = "-NoExit -Command `"cd '$WorkingDirectory'; $Command`""
    $processInfo.UseShellExecute = $true
    $processInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Normal
    
    $process = [System.Diagnostics.Process]::Start($processInfo)
    Write-Host "   ✅ Started (PID: $($process.Id))" -ForegroundColor Green
    Write-Host ""
    
    return $process
}

# Check if ports are available
Write-Host "🔍 Checking ports..." -ForegroundColor Cyan
Write-Host ""

if (Test-Port 8000) {
    Write-Host "⚠️  Port 8000 is already in use!" -ForegroundColor Yellow
    Write-Host "   The backend API might already be running." -ForegroundColor Yellow
    Write-Host ""
}

if (Test-Port 3000) {
    Write-Host "⚠️  Port 3000 is already in use!" -ForegroundColor Yellow
    Write-Host "   The frontend might already be running." -ForegroundColor Yellow
    Write-Host ""
}

# Start Backend API
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Step 1: Starting Backend API" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$backendProcess = Start-ProcessInWindow `
    -Title "LEAGAI Backend API" `
    -Command "uvicorn api.main:app --reload" `
    -WorkingDirectory $scriptDir

Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if backend is running
$maxAttempts = 10
$attempt = 0
$backendReady = $false

while ($attempt -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            break
        }
    }
    catch {
        # Backend not ready yet
    }
    
    $attempt++
    Start-Sleep -Seconds 1
}

if ($backendReady) {
    Write-Host "✅ Backend API is running at http://localhost:8000" -ForegroundColor Green
    Write-Host "✅ API Docs available at http://localhost:8000/docs" -ForegroundColor Green
}
else {
    Write-Host "⚠️  Backend might still be starting..." -ForegroundColor Yellow
}

Write-Host ""

# Create Test Accounts
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Step 2: Creating Test Accounts" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

if ($backendReady) {
    Write-Host "📌 Running: python create_test_accounts_api.py" -ForegroundColor Yellow
    & python create_test_accounts_api.py
    Write-Host ""
}
else {
    Write-Host "⚠️  Skipping test account creation (backend not ready)" -ForegroundColor Yellow
    Write-Host "   You can run it manually later: python create_test_accounts_api.py" -ForegroundColor Yellow
    Write-Host ""
}

# Start Frontend
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Step 3: Starting Frontend" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""

$frontendProcess = Start-ProcessInWindow `
    -Title "LEAGAI Frontend" `
    -Command "npm run dev" `
    -WorkingDirectory "$scriptDir\frontend"

Write-Host "⏳ Waiting for frontend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                  ✅ STARTUP COMPLETE!                         ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

Write-Host "🌐 Access the application:" -ForegroundColor Cyan
Write-Host "   Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "   Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host ""

Write-Host "🔐 Test Credentials:" -ForegroundColor Cyan
Write-Host "   Admin:     admin / AdminPass123!" -ForegroundColor White
Write-Host "   Agent:     agent1 / AgentPass123!" -ForegroundColor White
Write-Host "   Manager:   manager / ManagerPass123!" -ForegroundColor White
Write-Host "   Viewer:    viewer / ViewerPass123!" -ForegroundColor White
Write-Host ""

Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "   Quick Start:  QUICK_START.md" -ForegroundColor White
Write-Host "   Full Guide:   TEST_ACCOUNTS_GUIDE.md" -ForegroundColor White
Write-Host "   Color Theme:  DEEP_BLUE_COLOR_THEME_COMPLETE.md" -ForegroundColor White
Write-Host ""

Write-Host "💡 Tips:" -ForegroundColor Cyan
Write-Host "   • Check API docs at http://localhost:8000/docs" -ForegroundColor White
Write-Host "   • Use browser DevTools (F12) to debug frontend" -ForegroundColor White
Write-Host "   • Backend logs appear in the backend window" -ForegroundColor White
Write-Host "   • Frontend logs appear in the frontend window" -ForegroundColor White
Write-Host ""

Write-Host "🚀 Ready to develop! Happy coding!" -ForegroundColor Green
Write-Host ""

# Keep the script running
Write-Host "Press Ctrl+C to stop all services..." -ForegroundColor Yellow
while ($true) {
    Start-Sleep -Seconds 1
}

