#!/bin/bash

echo "🎵 Starting Music AI Recommendation System"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f env.example ]; then
        cp env.example .env
        echo "📝 Please edit .env file with your Spotify API credentials"
        echo "   You can get them from: https://developer.spotify.com/dashboard"
    else
        echo "❌ env.example file not found. Please create .env manually."
        exit 1
    fi
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  GPU not detected. Running on CPU (slower performance)"
    export CUDA_VISIBLE_DEVICES=""
else
    echo "🚀 GPU detected. Enabling CUDA acceleration"
    nvidia-smi --query-gpu=name --format=csv,noheader
fi

# Set environment variables
export PYTHONPATH=$PWD
export TOKENIZERS_PARALLELISM=false  # Avoid warnings

# Create necessary directories
mkdir -p models data logs

# Check if Redis is running
if ! redis-cli ping &> /dev/null; then
    echo "🔴 Starting Redis..."
    redis-server --daemonize yes
    sleep 2
fi

# Function to check if a service is running
check_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✅ $name is ready"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $name failed to start after $max_attempts attempts"
    return 1
}

# Start vLLM server with GPT-OSS-20B (if GPU available)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "🧠 Starting vLLM server with GPT-OSS-20B..."
    python -m vllm.entrypoints.api_server \
        --model openai/gpt-oss-20b \
        --host 0.0.0.0 \
        --port 8002 \
        --gpu-memory-utilization 0.8 \
        --max-model-len 4096 &
    
    VLLM_PID=$!
    
    # Wait for vLLM server to be ready
    if ! check_service "http://localhost:8002/health" "vLLM Server"; then
        echo "⚠️  vLLM server failed to start. Continuing without LLM explanations..."
        kill $VLLM_PID 2>/dev/null
        VLLM_PID=""
    fi
else
    echo "⚠️  Skipping vLLM server (no GPU available)"
    VLLM_PID=""
fi

# Start FastAPI backend
echo "⚡ Starting FastAPI backend..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to be ready
if ! check_service "http://localhost:8000/health" "Backend API"; then
    echo "❌ Backend failed to start"
    cleanup_and_exit 1
fi

# Start Celery worker for background tasks (optional)
echo "🔄 Starting background task worker..."
celery -A backend.tasks worker --loglevel=info &
CELERY_PID=$!

# Start Streamlit frontend
echo "🖥️  Starting Streamlit frontend..."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Wait for frontend to be ready
if ! check_service "http://localhost:8501/_stcore/health" "Streamlit Frontend"; then
    echo "❌ Frontend failed to start"
    cleanup_and_exit 1
fi

# Setup cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null
    [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ -n "$CELERY_PID" ] && kill $CELERY_PID 2>/dev/null
    [ -n "$STREAMLIT_PID" ] && kill $STREAMLIT_PID 2>/dev/null
    redis-cli shutdown 2>/dev/null
    echo "👋 Cleanup complete"
}

cleanup_and_exit() {
    cleanup
    exit $1
}

trap cleanup_and_exit EXIT INT TERM

echo ""
echo "🎉 Music AI Recommendation System is now running!"
echo ""
echo "🖥️  Frontend: http://localhost:8501"
echo "⚡ Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
if [ -n "$VLLM_PID" ]; then
    echo "🧠 vLLM Server: http://localhost:8002"
fi
echo "🔄 Celery Monitor: http://localhost:5555 (if available)"
echo ""
echo "Press Ctrl+C to stop all services"

# Keep script running
wait
