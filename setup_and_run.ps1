# LEAGAI Complete Setup and Run Script
# This script sets up and runs the entire LEAGAI application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 LEAGAI Complete Setup & Run Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "✓ Checking Python installation..." -ForegroundColor Green
python --version
Write-Host ""

# Step 2: Install dependencies
Write-Host "✓ Installing dependencies..." -ForegroundColor Green
pip install -r requirements-minimal.txt --no-cache-dir -q
Write-Host "✓ Dependencies installed!" -ForegroundColor Green
Write-Host ""

# Step 3: Start Backend
Write-Host "✓ Starting Backend API on http://127.0.0.1:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd api; python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000" -WindowStyle Normal
Write-Host "✓ Backend started in new window!" -ForegroundColor Green
Write-Host ""

# Step 4: Wait for backend to start
Write-Host "⏳ Waiting for backend to start (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10
Write-Host ""

# Step 5: Start Frontend
Write-Host "✓ Starting Frontend on http://localhost:3000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev" -WindowStyle Normal
Write-Host "✓ Frontend started in new window!" -ForegroundColor Green
Write-Host ""

# Step 6: Create Test Accounts
Write-Host "✓ Creating test accounts..." -ForegroundColor Green
Start-Sleep -Seconds 5
python create_test_accounts_api.py
Write-Host ""

# Step 7: Display Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ LEAGAI Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host "🔌 Backend API: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "📚 API Docs: http://127.0.0.1:8000/v1/docs" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Test Accounts:" -ForegroundColor Yellow
Write-Host "  Admin:   admin@leagai.dev / AdminPass123!" -ForegroundColor White
Write-Host "  Manager: manager@leagai.dev / ManagerPass123!" -ForegroundColor White
Write-Host "  Agent1:  agent1@leagai.dev / AgentPass123!" -ForegroundColor White
Write-Host "  Agent2:  agent2@leagai.dev / AgentPass456!" -ForegroundColor White
Write-Host "  Viewer:  viewer@leagai.dev / ViewerPass123!" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Open http://localhost:3000/login in your browser" -ForegroundColor White
Write-Host "  2. Login with any test account above" -ForegroundColor White
Write-Host "  3. Explore the dashboard and features" -ForegroundColor White
Write-Host ""
Write-Host "📖 Documentation:" -ForegroundColor Cyan
Write-Host "  - Backend API Docs: http://127.0.0.1:8000/v1/docs" -ForegroundColor White
Write-Host "  - Backend ReDoc: http://127.0.0.1:8000/v1/redoc" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

