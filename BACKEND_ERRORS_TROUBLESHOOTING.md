# 🔧 Backend Errors Troubleshooting Guide

This guide addresses the common warnings and errors that appear when starting the LEGEAI backend.

---

## 📋 Error Summary

The errors fall into **3 categories**:

1. **Redis Connection Error** (Optional service)
2. **ML Model Files Missing** (Non-critical - has fallbacks)
3. **Database Not Initialized** (Optional - has in-memory fallback)

**IMPORTANT:** ✅ **These errors are NON-CRITICAL**. The application will run with fallback mechanisms.

---

## 🔴 Error 1: Redis Connection Refused

### Error Message:
```
Redis unavailable, using in-memory only: Error 10061 connecting to localhost:6379. 
No connection could be made because the target machine actively refused it.
```

### What This Means:
- Redis server is not running on your machine
- The application will use **in-memory caching** instead
- **No functionality is lost** - just slower cache performance

### Solutions:

#### Option 1: Ignore It (Recommended for Development)
**The application works fine without Redis.** No action needed.

#### Option 2: Install and Start Redis (Optional)

**Windows:**
```powershell
# Download Redis for Windows from:
# https://github.com/microsoftarchive/redis/releases

# Or use WSL2:
wsl --install
wsl
sudo apt update
sudo apt install redis-server
sudo service redis-server start
```

**macOS:**
```bash
# Install via Homebrew
brew install redis

# Start Redis
brew services start redis

# Verify it's running
redis-cli ping  # Should return "PONG"
```

**Linux:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### Option 3: Use Docker
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

---

## 🔴 Error 2: ML Model Files Missing

### Error Messages:
```
ERROR: Error loading life insurance model: [Errno 2] No such file or directory: 
'models/life_insurance_scoring/artifacts/model.pkl'

ERROR: Failed to load auto insurance deep learning model: [Errno 2] No such file or directory:
'backend/models/auto_insurance_scoring/saved_models/auto_insurance_dl_scaler.pkl'

ERROR: Error loading Home Insurance Deep Learning model: [Errno 2] No such file or directory:
'backend/models/home_insurance_scoring/saved_models/home_insurance_dl_features.pkl'

ERROR: Error loading Health Insurance Deep Learning model: [Errno 2] No such file or directory:
'backend/models/healthcare_insurance_scoring/saved_models/health_insurance_dl_features.pkl'
```

### What This Means:
- Pre-trained ML models are not present in the repository
- The application will use **rule-based scoring fallback**
- **Lead scoring still works** - just uses simpler algorithms

### Solutions:

#### Option 1: Use Rule-Based Fallback (Recommended for Development)
**No action needed.** The application automatically falls back to rule-based scoring.

**How it works:**
- Uses business logic instead of ML models
- Scores leads based on predefined rules
- Fully functional for testing and development

#### Option 2: Train the Models (For Production)

**Train all models:**
```bash
# From project root
cd backend

# Train insurance lead scoring model
python -m models.insurance_lead_scoring.train

# Train life insurance model
python -m models.life_insurance_scoring.train

# Train auto insurance deep learning model
python scripts/train_auto_insurance_dl.py

# Train home insurance model
python -m models.home_insurance_scoring.train_deep_learning

# Train health insurance model
python -m models.healthcare_insurance_scoring.train_deep_learning
```

**Note:** Training requires:
- Training data in `data/` directory
- Python packages: `scikit-learn`, `xgboost`, `torch`, `pandas`, `numpy`
- 5-10 minutes per model

---

## 🔴 Error 3: Database Not Initialized

### Error Message:
```
WARNING: DB unavailable or not initialized: database "leadgen" does not exist
```

### What This Means:
- PostgreSQL database hasn't been created
- The application will use **in-memory storage**
- **All features work** - data is just not persisted

### Solutions:

#### Option 1: Use In-Memory Storage (Recommended for Development)

