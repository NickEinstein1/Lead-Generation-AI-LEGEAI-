# Train All ML Models - Windows PowerShell Script
# This script trains all insurance lead scoring models

Write-Host "🤖 LEGEAI - Train All ML Models" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Set PYTHONPATH
$env:PYTHONPATH = "."
Write-Host "✅ PYTHONPATH set to current directory" -ForegroundColor Green
Write-Host ""

# Check if training data exists
Write-Host "📊 Checking training data..." -ForegroundColor Yellow
if (-not (Test-Path "data/insurance_leads_training.csv")) {
    Write-Host "⚠️  Training data not found. Generating..." -ForegroundColor Yellow
    python generate_training_data.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to generate training data" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Training data generated" -ForegroundColor Green
} else {
    Write-Host "✅ Training data found" -ForegroundColor Green
}
Write-Host ""

# Train Insurance Lead Scoring (XGBoost)
Write-Host "1️⃣  Training Insurance Lead Scoring (XGBoost)..." -ForegroundColor Cyan
python backend/models/insurance_lead_scoring/train.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Insurance Lead Scoring model trained" -ForegroundColor Green
} else {
    Write-Host "⚠️  Insurance Lead Scoring training failed (continuing...)" -ForegroundColor Yellow
}
Write-Host ""

# Train Life Insurance XGBoost
Write-Host "2️⃣  Training Life Insurance (XGBoost)..." -ForegroundColor Cyan
python backend/models/life_insurance_scoring/train.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Life Insurance XGBoost model trained" -ForegroundColor Green
} else {
    Write-Host "⚠️  Life Insurance XGBoost training failed (continuing...)" -ForegroundColor Yellow
}
Write-Host ""

# Train Life Insurance Deep Learning
Write-Host "3️⃣  Training Life Insurance (Deep Learning)..." -ForegroundColor Cyan
python backend/models/life_insurance_scoring/train_deep_learning.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Life Insurance Deep Learning model trained" -ForegroundColor Green
} else {
    Write-Host "⚠️  Life Insurance Deep Learning training failed (continuing...)" -ForegroundColor Yellow
}
Write-Host ""

# Train Auto Insurance Deep Learning
Write-Host "4️⃣  Training Auto Insurance (Deep Learning)..." -ForegroundColor Cyan
python backend/models/auto_insurance_scoring/train_deep_learning.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Auto Insurance Deep Learning model trained" -ForegroundColor Green
} else {
    Write-Host "⚠️  Auto Insurance Deep Learning training failed (continuing...)" -ForegroundColor Yellow
}
Write-Host ""

# Train Home Insurance Deep Learning
Write-Host "5️⃣  Training Home Insurance (Deep Learning)..." -ForegroundColor Cyan
python backend/models/home_insurance_scoring/train_deep_learning.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Home Insurance Deep Learning model trained" -ForegroundColor Green
} else {
    Write-Host "⚠️  Home Insurance Deep Learning training failed (continuing...)" -ForegroundColor Yellow
}
Write-Host ""

# Train Health Insurance Deep Learning
Write-Host "6️⃣  Training Health Insurance (Deep Learning)..." -ForegroundColor Cyan
python backend/models/healthcare_insurance_scoring/train_deep_learning.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Health Insurance Deep Learning model trained" -ForegroundColor Green
} else {
    Write-Host "⚠️  Health Insurance Deep Learning training failed (continuing...)" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "🎉 Training Complete!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📁 Model files saved to:" -ForegroundColor Yellow
Write-Host "   - models/insurance_lead_scoring/artifacts/" -ForegroundColor White
Write-Host "   - models/life_insurance_scoring/artifacts/" -ForegroundColor White
Write-Host "   - models/life_insurance_scoring/deep_learning_artifacts/" -ForegroundColor White
Write-Host "   - backend/models/auto_insurance_scoring/saved_models/" -ForegroundColor White
Write-Host "   - backend/models/home_insurance_scoring/saved_models/" -ForegroundColor White
Write-Host "   - backend/models/healthcare_insurance_scoring/saved_models/" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Start the backend: .\run_backend.ps1" -ForegroundColor White
Write-Host "   2. Verify no model loading errors in logs" -ForegroundColor White
Write-Host "   3. Test lead scoring endpoints" -ForegroundColor White
Write-Host ""

