#!/bin/bash

# AI Music Discovery Engine - Startup Script
# Starts both FastAPI backend and Streamlit frontend

set -e

echo "============================================================"
echo "🎵 AI Music Discovery Engine - Starting Services"
echo "============================================================"

# Load environment variables
source .env

# Check if credentials exist
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ Error: credentials.json not found"
    echo "   Please ensure your Google Cloud credentials are in place"
    exit 1
fi

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "streamlit.*app.py" 2>/dev/null || true
sleep 2

# Start FastAPI backend
echo ""
echo "🚀 Starting FastAPI backend on port 8080..."
nohup python main.py > api.log 2>&1 &
API_PID=$!
echo "   Backend PID: $API_PID"

# Wait for API to start
echo "   Waiting for API to be ready..."
for i in {1..10}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "   ✅ Backend is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "   ❌ Backend failed to start. Check api.log"
        tail -n 20 api.log
        exit 1
    fi
    sleep 1
    echo -n "."
done

# Start Streamlit frontend
echo ""
echo "🎨 Starting Streamlit frontend on port 8501..."
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
UI_PID=$!
echo "   Frontend PID: $UI_PID"

# Wait for Streamlit to start
echo "   Waiting for frontend to be ready..."
sleep 3

echo ""
echo "============================================================"
echo "✅ Services are running!"
echo "============================================================"
echo ""
echo "📡 Backend API:  http://localhost:8080"
echo "   Health check: curl http://localhost:8080/health"
echo "   Logs:         tail -f api.log"
echo ""
echo "🎨 Frontend UI:  http://localhost:8501"
echo "   Logs:         tail -f streamlit.log"
echo ""
echo "============================================================"
echo "🛑 To stop services:"
echo "   pkill -f 'python.*main.py'"
echo "   pkill -f 'streamlit.*app.py'"
echo "============================================================"
echo ""
echo "🎵 Happy music discovering! 🚀"
echo ""
