# 🪟 Running LEGEAI Backend on Windows

## 🚀 Quick Start - 3 Methods

---

## **Method 1: Using Batch Script (Easiest)** ⭐

### Step 1: Open Command Prompt
- Press `Win + R`
- Type `cmd` and press Enter

### Step 2: Navigate to Project
```cmd
cd C:\path\to\Lead-Generation-AI-LEGEAI-
```

### Step 3: Run the Script
```cmd
start_backend.bat
```

**That's it!** The backend will start automatically.

---

## **Method 2: Using PowerShell Script** ⭐

### Step 1: Open PowerShell
- Press `Win + X`
- Select "Windows PowerShell" or "Terminal"

### Step 2: Navigate to Project
```powershell
cd C:\path\to\Lead-Generation-AI-LEGEAI-
```

### Step 3: Allow Script Execution (First Time Only)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Run the Script
```powershell
.\start_backend.ps1
```

---

## **Method 3: Manual Commands**

### Using Command Prompt (cmd):

```cmd
REM Navigate to project
cd C:\path\to\Lead-Generation-AI-LEGEAI-

REM Activate virtual environment
.venv\Scripts\activate.bat

REM Start backend
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Using PowerShell:

```powershell
# Navigate to project
cd C:\path\to\Lead-Generation-AI-LEGEAI-

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Start backend
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

---

## ✅ **What You Should See**

When the backend starts successfully:

```
INFO:     Will watch for changes in these directories: ['C:\\path\\to\\Lead-Generation-AI-LEGEAI-']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## 🌐 **Verify It's Running**

Open your browser and visit:

- **API Documentation**: http://localhost:8000/v1/docs
- **Health Check**: http://localhost:8000/health
- **Meta Integration**: http://localhost:8000/v1/integrations/meta/status

---

## 🐛 **Troubleshooting**

### ❌ Error: "python is not recognized"

**Solution**: Install Python or add it to PATH

1. Download Python from https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Restart Command Prompt/PowerShell

### ❌ Error: "No module named 'uvicorn'"

**Solution**: Install dependencies

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### ❌ Error: "Virtual environment not found"

**Solution**: Create virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### ❌ Error: "Address already in use" (Port 8000)

**Solution**: Kill process using port 8000

```cmd
REM Find process using port 8000
netstat -ano | findstr :8000

REM Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### ❌ Error: "cannot be loaded because running scripts is disabled"

**Solution**: Enable PowerShell scripts

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📦 **First Time Setup**

If this is your first time running the backend:

### 1. Create Virtual Environment
```cmd
python -m venv .venv
```

### 2. Activate Virtual Environment
```cmd
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root with:

```env
# Backend
USE_DB=true
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/leadgen
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=change-me

# Meta Marketing API Integration
META_APP_ID=your_app_id_here
META_APP_SECRET=your_app_secret_here
META_REDIRECT_URI=http://localhost:8000/v1/integrations/meta/callback
META_VERIFY_TOKEN=legeai_meta_webhook_2024
META_API_VERSION=v21.0

# Frontend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1
```

### 5. Run Database Migration
```cmd
cd backend
alembic upgrade head
cd ..
```

### 6. Start Backend
```cmd
start_backend.bat
```

---

## 🎯 **Running Both Backend and Frontend**

### Terminal 1 - Backend:
```cmd
start_backend.bat
```

### Terminal 2 - Frontend:
```cmd
cd frontend
npm install
npm run dev
```

Then open: http://localhost:3000

---

## 📝 **Quick Reference**

| Task | Command (cmd) | Command (PowerShell) |
|------|---------------|---------------------|
| Activate venv | `.venv\Scripts\activate.bat` | `.venv\Scripts\Activate.ps1` |
| Deactivate venv | `deactivate` | `deactivate` |
| Install deps | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Start backend | `start_backend.bat` | `.\start_backend.ps1` |
| Check port 8000 | `netstat -ano \| findstr :8000` | `netstat -ano \| findstr :8000` |
| Kill process | `taskkill /PID <PID> /F` | `Stop-Process -Id <PID>` |

---

## 🎉 **Success!**

Once the backend is running, you can:

1. ✅ Access API docs at http://localhost:8000/v1/docs
2. ✅ Test Meta integration endpoints
3. ✅ Start the frontend and connect to backend
4. ✅ Begin capturing leads from Meta!

---

**Need help?** Check the main documentation:
- `RUN_META_INTEGRATION.md` - Complete Meta setup guide
- `docs/META_QUICK_START.md` - Quick start guide
- `docs/META_END_TO_END_TESTING.md` - Testing guide

