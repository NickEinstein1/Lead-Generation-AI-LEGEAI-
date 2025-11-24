#!/bin/bash

# LEGEAI - Automated Setup Script for macOS/Linux
# This script automates the installation and setup process

set -e  # Exit on error

echo "🚀 LEGEAI - Automated Setup Script"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Python is installed
print_info "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Check if Node.js is installed
print_info "Checking Node.js installation..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION found"
else
    print_error "Node.js not found. Please install Node.js 18.17 or higher."
    exit 1
fi

# Check if npm is installed
print_info "Checking npm installation..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_success "npm $NPM_VERSION found"
else
    print_error "npm not found. Please install npm."
    exit 1
fi

echo ""
print_info "Step 1: Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

echo ""
print_info "Step 2: Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

echo ""
print_info "Step 3: Upgrading pip..."
pip install --upgrade pip --quiet
print_success "pip upgraded"

echo ""
print_info "Step 4: Installing Python dependencies..."
print_warning "This may take 5-10 minutes..."
pip install -r requirements.txt --quiet
print_success "Python dependencies installed"

echo ""
print_info "Step 5: Creating .env file..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
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
EOF
    print_success ".env file created"
else
    print_warning ".env file already exists"
fi

echo ""
print_info "Step 6: Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    print_warning "This may take 3-5 minutes..."
    npm install --silent
    print_success "Frontend dependencies installed"
else
    print_warning "node_modules already exists, running npm install to update..."
    npm install --silent
    print_success "Frontend dependencies updated"
fi

echo ""
print_info "Step 7: Creating frontend .env.local file..."
if [ ! -f ".env.local" ]; then
    cat > .env.local << EOF
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Environment
NODE_ENV=development
EOF
    print_success "Frontend .env.local file created"
else
    print_warning "Frontend .env.local file already exists"
fi

cd ..

echo ""
echo "=================================="
print_success "Setup completed successfully!"
echo "=================================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Start the backend (Terminal 1):"
echo "   ${BLUE}source venv/bin/activate${NC}"
echo "   ${BLUE}python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000${NC}"
echo ""
echo "2. Start the frontend (Terminal 2):"
echo "   ${BLUE}cd frontend${NC}"
echo "   ${BLUE}npm run dev${NC}"
echo ""
echo "3. Open your browser:"
echo "   Frontend: ${GREEN}http://localhost:3000${NC}"
echo "   API Docs: ${GREEN}http://localhost:8000/v1/docs${NC}"
echo ""
echo "4. Login with test credentials:"
echo "   Email: ${YELLOW}admin@legeai.dev${NC}"
echo "   Password: ${YELLOW}AdminPass123!${NC}"
echo ""
print_info "For more information, see SETUP_GUIDE.md"
echo ""

