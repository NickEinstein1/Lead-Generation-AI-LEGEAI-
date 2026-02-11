#!/bin/bash

# Start Backend Server for LEGEAI
# This script activates the virtual environment and starts the FastAPI backend

echo "🚀 Starting LEGEAI Backend Server..."
echo ""

# Activate virtual environment
source .venv/bin/activate

# Check if uvicorn is installed
if ! python -m uvicorn --version &> /dev/null; then
    echo "❌ Error: uvicorn not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the backend server
echo "✅ Starting FastAPI server on http://127.0.0.1:8000"
echo "📚 API Documentation: http://127.0.0.1:8000/v1/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

