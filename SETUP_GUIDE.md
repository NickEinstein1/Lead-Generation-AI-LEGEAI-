# 🚀 LEGEAI - Complete Setup & Installation Guide

This guide provides step-by-step instructions for setting up, installing, and running the LEGEAI platform on your local machine or server.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Project Setup](#project-setup)
4. [Backend Setup](#backend-setup)
5. [Frontend Setup](#frontend-setup)
6. [Database Setup](#database-setup)
7. [Running the Application](#running-the-application)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **CPU**: 4 cores recommended
- **Internet**: Stable connection for package downloads

### Software Requirements
- **Python**: 3.10 or 3.11 (3.10 recommended)
- **Node.js**: 18.17.0 or higher (20.x LTS recommended)
- **npm**: 9.0.0 or higher (comes with Node.js)
- **Git**: Latest version
- **PostgreSQL**: 15.x (optional - can use in-memory fallback)
- **Redis**: 7.x (optional - can use in-memory fallback)

---

## 🔧 Prerequisites Installation

### 1. Install Python

**Windows:**
```powershell
# Download from https://www.python.org/downloads/
# OR use winget
winget install Python.Python.3.10

# Verify installation
python --version
# Should output: Python 3.10.x
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.10

# Verify installation
python3 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Verify installation
python3 --version
```

### 2. Install Node.js & npm

**Windows:**
```powershell
# Download from https://nodejs.org/
# OR use winget
winget install OpenJS.NodeJS.LTS

# Verify installation
node --version
npm --version
```

**macOS:**
```bash
# Using Homebrew
brew install node@20

# Verify installation
node --version
npm --version
```

**Linux (Ubuntu/Debian):**
```bash
# Using NodeSource repository
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

### 3. Install Git

**Windows:**
```powershell
winget install Git.Git
```

**macOS:**
```bash
brew install git
```

**Linux:**
```bash
sudo apt install git
```

### 4. Install PostgreSQL (Optional)

**Windows:**
```powershell
# Download from https://www.postgresql.org/download/windows/
# OR use winget
winget install PostgreSQL.PostgreSQL.15
```

**macOS:**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Linux:**
```bash
sudo apt install postgresql-15 postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 5. Install Redis (Optional)

**Windows:**
```powershell
# Download from https://github.com/microsoftarchive/redis/releases
# OR use WSL and install Redis in Linux
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Linux:**
```bash
sudo apt install redis-server
sudo systemctl start redis
sudo systemctl enable redis
```

---

## 📦 Project Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/Lead-Generation-AI-LEGEAI-.git

# Navigate to project directory
cd Lead-Generation-AI-LEGEAI-

# Verify you're in the correct directory
ls
# You should see: backend/, frontend/, requirements.txt, README.md, etc. (optional)
```

## 🐍 Backend Setup

### 1. Create Python Virtual Environment

**Windows (PowerShell):**
```powershell
# Navigate to project root
cd Lead-Generation-AI-LEGEAI-

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify activation (you should see (venv) in prompt)
```

**macOS/Linux (Bash):**
```bash
# Navigate to project root
cd Lead-Generation-AI-LEGEAI-

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in prompt)
```

### 2. Install Python Dependencies

```bash
# Make sure virtual environment is activated (you should see (venv) in prompt)

# Upgrade pip
pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# This will install 100+ packages including:
# - FastAPI 0.104.1
# - SQLAlchemy 2.0.44
# - Uvicorn 0.24.0
# - Pydantic 2.7.0+
# - PyJWT 2.10.1
# - bcrypt 4.2.0
# - pandas, numpy, scikit-learn
# - xgboost, lightgbm, catboost
# - And many more...

# Installation may take 5-10 minutes depending on your internet speed
```

### 3. Set Environment Variables

**Create `.env` file in project root:**

```bash
# Windows
New-Item -Path .env -ItemType File

# macOS/Linux
touch .env
```

**Add the following to `.env`:**

```env
# Database Configuration
USE_DB=false
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/legeai

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# JWT Secret (change in production!)
JWT_SECRET_KEY=your-super-secret-key-change-in-production

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# DocuSeal Integration (optional)
DOCUSEAL_API_KEY=your-docuseal-api-key
DOCUSEAL_TEMPLATE_ID=your-template-id
DOCUSEAL_FORM_BASE=https://docuseal.com
DOCUSEAL_API_BASE=https://api.docuseal.com

# CORS Origins
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### 4. Initialize Database (Optional)

**If using PostgreSQL:**

```bash
# Create database
createdb legeai

# OR using psql
psql -U postgres
CREATE DATABASE legeai;
\q

# Run migrations
cd backend
alembic upgrade head
cd ..
```

**If NOT using PostgreSQL:**
- Set `USE_DB=false` in `.env`
- The application will use in-memory storage (data will be lost on restart)

### 5. Verify Backend Installation

```bash
# Check Python packages
pip list | grep -E "fastapi|uvicorn|sqlalchemy"

# Should show:
# fastapi         0.104.1
# uvicorn         0.24.0
# SQLAlchemy      2.0.44
```

---

## ⚛️ Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd frontend
```

### 2. Install Node Dependencies

```bash
# Install all dependencies from package.json
npm install

# This will install:
# - next 15.5.5
# - react 19.1.0
# - react-dom 19.1.0
# - typescript 5.x
# - tailwindcss 4.x
# - recharts 3.4.1
# - And more...

# Installation may take 3-5 minutes
```

**Alternative: Using Yarn**
```bash
# If you prefer yarn
yarn install
```

### 3. Verify Frontend Installation

```bash
# Check installed packages
npm list --depth=0

# Should show all dependencies from package.json
```

### 4. Configure Frontend Environment

**Create `.env.local` in `frontend/` directory:**

```bash
# Create file
touch .env.local  # macOS/Linux
New-Item -Path .env.local -ItemType File  # Windows
```

**Add the following to `frontend/.env.local`:**

```env
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Environment
NODE_ENV=development
```

### 5. Return to Project Root

```bash
cd ..
```

---

## 🗄️ Database Setup

### Option 1: Using In-Memory Storage (Easiest)

**No setup required!**
- Set `USE_DB=false` in `.env`
- Data stored in memory (lost on restart)
- Perfect for development and testing

### Option 2: Using PostgreSQL (Recommended for Production)

**1. Create Database:**

```bash
# Using createdb command
createdb -U postgres legeai

# OR using psql
psql -U postgres
CREATE DATABASE legeai;
\q
```

**2. Update `.env` file:**

```env
USE_DB=true
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/legeai
```

**3. Run Database Migrations:**

```bash
# Navigate to backend directory
cd backend

# Run all migrations
alembic upgrade head

# You should see output like:
# INFO  [alembic.runtime.migration] Running upgrade -> abc123, initial
# INFO  [alembic.runtime.migration] Running upgrade abc123 -> def456, add_users
# ...

# Return to project root
cd ..
```

**4. Create Test Accounts (Optional):**

```bash
# Run the test account creation script
python create_test_accounts_api.py

# This creates 5 test accounts:
# - admin@legeai.dev (Admin)
# - manager@legeai.dev (Manager)
# - agent1@legeai.dev (Agent)
# - agent2@legeai.dev (Agent)
# - viewer@legeai.dev (Viewer)
```

### Option 3: Using Docker (PostgreSQL + Redis)

```bash
# Start PostgreSQL and Redis using Docker Compose
docker-compose up -d db redis

# Verify containers are running
docker ps

# Update .env
USE_DB=true
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/leadgen

# Run migrations
cd backend
alembic upgrade head
cd ..
```

---

## 🚀 Running the Application

### Method 1: Using PowerShell Scripts (Windows - Recommended)

**Start Backend:**
```powershell
# From project root
.\run_backend.ps1

# You should see:
# 🚀 Starting LEGEAI Backend...
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete.
```

**Start Frontend (in a new terminal):**
```powershell
cd frontend
npm run dev

# You should see:
# ▲ Next.js 15.5.5
# - Local:        http://localhost:3000
# ✓ Ready in 2.5s
```

### Method 2: Manual Start (All Platforms)

**Terminal 1 - Backend:**
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows

# Set environment variables
export PYTHONPATH=.  # macOS/Linux
$env:PYTHONPATH = "."  # Windows PowerShell

export USE_DB=false  # macOS/Linux
$env:USE_DB = "false"  # Windows PowerShell

# Start backend
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

# Backend will be available at: http://localhost:8000
# API docs at: http://localhost:8000/v1/docs
```

**Terminal 2 - Frontend:**
```bash
# Navigate to frontend
cd frontend

# Start development server
npm run dev

# Frontend will be available at: http://localhost:3000
```

### Method 3: Using Docker Compose (Full Stack)

```bash
# Start all services (backend, frontend, database, redis)
docker-compose up

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Verify Installation

**1. Check Backend:**
- Open browser: http://localhost:8000/v1/docs
- You should see Swagger UI with API documentation

**2. Check Frontend:**
- Open browser: http://localhost:3000
- You should see the futuristic LEGEAI landing page

**3. Test Login:**
- Click "Login" button
- Use test credentials:
  - Email: `admin@legeai.dev`
  - Password: `AdminPass123!`
- You should be redirected to the dashboard

---

## 🔧 Troubleshooting

### Backend Issues

**Issue: Port 8000 already in use**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

**Issue: Module not found errors**
```bash
# Make sure PYTHONPATH is set
export PYTHONPATH=.  # macOS/Linux
$env:PYTHONPATH = "."  # Windows

# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows
```

**Issue: Database connection errors**
```env
# Use in-memory storage instead
USE_DB=false
```

**Issue: Import errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Frontend Issues

**Issue: Port 3000 already in use**
```bash
# Use a different port
npm run dev -- -p 3001
```

**Issue: Dependencies not installing**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Issue: TypeScript errors**
```bash
# Rebuild TypeScript
npm run build
```

**Issue: Module not found**
```bash
# Clear Next.js cache
rm -rf .next
npm run dev
```

### Common Issues

**Issue: Python version mismatch**
```bash
# Check Python version
python --version

# Should be 3.9, 3.10, or 3.11
# If not, install correct version
```

**Issue: Node version mismatch**
```bash
# Check Node version
node --version

# Should be 18.17.0 or higher
# If not, install correct version
```

---

## 🔄 Updating the Application

### Update Backend

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows

# Pull latest changes
git pull origin main

# Update Python dependencies
pip install --upgrade -r requirements.txt

# Run new migrations (if any)
cd backend
alembic upgrade head
cd ..

# Restart backend
# Stop current backend (Ctrl+C)
# Start again with run_backend.ps1 or manual command
```

### Update Frontend

```bash
# Navigate to frontend
cd frontend

# Pull latest changes (if not already done)
git pull origin main

# Update Node dependencies
npm install

# Clear cache
rm -rf .next

# Restart frontend
# Stop current frontend (Ctrl+C)
npm run dev
```

### Update Database Schema

```bash
# If there are new migrations
cd backend
alembic upgrade head
cd ..

# To rollback a migration
cd backend
alembic downgrade -1
cd ..

# To see migration history
cd backend
alembic history
cd ..
```

---

## 🌐 Production Deployment

### Environment Variables for Production

**Backend `.env`:**
```env
# Database - Use production database
USE_DB=true
DATABASE_URL=postgresql+asyncpg://user:password@production-db:5432/legeai

# Redis - Use production Redis
REDIS_URL=redis://production-redis:6379/0

# JWT Secret - Use strong secret!
JWT_SECRET_KEY=your-very-strong-production-secret-key-min-32-chars

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# CORS - Add your production domains
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# DocuSeal
DOCUSEAL_API_KEY=your-production-api-key
DOCUSEAL_TEMPLATE_ID=your-template-id
```

**Frontend `.env.production`:**
```env
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NODE_ENV=production
```

### Build for Production

**Backend:**
```bash
# No build step required for FastAPI
# Just ensure all dependencies are installed
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend

# Build production bundle
npm run build

# Start production server
npm start

# Or use PM2 for process management
npm install -g pm2
pm2 start npm --name "legeai-frontend" -- start
```

### Using Docker for Production

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale services
docker-compose up -d --scale api=3

# Stop services
docker-compose down
```

### Nginx Configuration (Optional)

**`/etc/nginx/sites-available/legeai`:**
```nginx
# Frontend
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# Backend API
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 📊 System Paths Reference

### Important File Paths

```
Project Root: /Users/einstein/Documents/Projects/Lead-Generation-AI-LEGEAI-

Backend:
├── Entry Point:        backend/api/main.py
├── Models:             backend/models/
├── API Endpoints:      backend/api/
├── Database Config:    backend/database/
├── Migrations:         backend/alembic/versions/
├── Alembic Config:     backend/alembic.ini
└── Requirements:       requirements.txt

Frontend:
├── Entry Point:        frontend/src/app/page.tsx
├── App Router:         frontend/src/app/
├── Components:         frontend/src/components/
├── Lib/Utils:          frontend/src/lib/
├── Public Assets:      frontend/public/
├── Dependencies:       frontend/package.json
└── TypeScript Config:  frontend/tsconfig.json

Configuration:
├── Environment:        .env (create this)
├── Docker Compose:     docker-compose.yml
├── Backend Script:     run_backend.ps1
└── Documentation:      README.md, SETUP_GUIDE.md

Data:
├── Training Data:      data/*.csv
├── ML Models:          models/
└── Uploads:            uploads/
```

### Port Assignments

```
Frontend:           http://localhost:3000
Backend API:        http://localhost:8000
API Docs (Swagger): http://localhost:8000/v1/docs
API Docs (ReDoc):   http://localhost:8000/v1/redoc
PostgreSQL:         localhost:5432
Redis:              localhost:6379
```

### Virtual Environment Paths

```
Windows:
├── Activate:   .\venv\Scripts\Activate.ps1
├── Deactivate: deactivate
└── Python:     .\venv\Scripts\python.exe

macOS/Linux:
├── Activate:   source venv/bin/activate
├── Deactivate: deactivate
└── Python:     ./venv/bin/python
```

---

## 📝 Quick Reference Commands

### Daily Development Workflow

```bash
# 1. Activate virtual environment
source venv/bin/activate  # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows

# 2. Start backend (Terminal 1)
.\run_backend.ps1  # Windows
# OR
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

# 3. Start frontend (Terminal 2)
cd frontend
npm run dev

# 4. Open browser
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/v1/docs
```

### Useful Commands

```bash
# Check running processes
lsof -i :8000  # Backend (macOS/Linux)
lsof -i :3000  # Frontend (macOS/Linux)
netstat -ano | findstr :8000  # Backend (Windows)
netstat -ano | findstr :3000  # Frontend (Windows)

# Kill processes
kill -9 $(lsof -ti:8000)  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# View logs
tail -f backend/logs/app.log  # Backend logs (if configured)
docker-compose logs -f  # Docker logs

# Database operations
alembic upgrade head  # Apply migrations
alembic downgrade -1  # Rollback one migration
alembic history  # View migration history
alembic current  # View current migration

# Clear caches
rm -rf frontend/.next  # Next.js cache
rm -rf frontend/node_modules  # Node modules
rm -rf __pycache__  # Python cache
```

---

## ✅ Installation Checklist

Use this checklist to ensure everything is set up correctly:

- [ ] Python 3.9+ installed and verified
- [ ] Node.js 18.17+ installed and verified
- [ ] Git installed
- [ ] PostgreSQL installed (optional)
- [ ] Redis installed (optional)
- [ ] Repository cloned
- [ ] Python virtual environment created
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file created in project root
- [ ] Frontend dependencies installed (`npm install`)
- [ ] `.env.local` file created in `frontend/`
- [ ] Database created (if using PostgreSQL)
- [ ] Migrations run (`alembic upgrade head`)
- [ ] Test accounts created (optional)
- [ ] Backend starts successfully on port 8000
- [ ] Frontend starts successfully on port 3000
- [ ] Can access API docs at http://localhost:8000/v1/docs
- [ ] Can access landing page at http://localhost:3000
- [ ] Can login with test credentials
- [ ] Dashboard loads successfully

---

## 🆘 Getting Help

### Resources

- **Documentation**: See `README.md` and `/docs` folder
- **API Documentation**: http://localhost:8000/v1/docs
- **GitHub Issues**: Report bugs and request features
- **Email Support**: support@legeai.dev

### Common Questions

**Q: Do I need PostgreSQL and Redis?**
A: No, you can use in-memory storage by setting `USE_DB=false` in `.env`. This is perfect for development.

**Q: Which Python version should I use?**
A: Python 3.10 is recommended. Python 3.9 and 3.11 also work.

**Q: Can I use yarn instead of npm?**
A: Yes, yarn is fully supported for the frontend.

**Q: How do I reset the database?**
A: Run `alembic downgrade base` then `alembic upgrade head` to reset all migrations.

**Q: Where are the logs stored?**
A: Backend logs are printed to console. Configure file logging in `backend/config/` if needed.

---

## 🎉 Success!

If you've completed all steps, you should now have:

✅ **Backend running** on http://localhost:8000
✅ **Frontend running** on http://localhost:3000
✅ **API documentation** accessible
✅ **Database** configured (or in-memory fallback)
✅ **Test accounts** ready to use

**Next Steps:**
1. Explore the dashboard at http://localhost:3000/dashboard
2. Test the API at http://localhost:8000/v1/docs
3. Create your first lead
4. Customize the application for your needs

---

**Made with ❤️ by the LEGEAI Team**

**Last Updated**: November 2025
**Version**: 1.0.0
