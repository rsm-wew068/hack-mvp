#!/usr/bin/env bash
set -euo pipefail

# Best-effort: free common ports
fuser -k 8002/tcp 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 7860/tcp 2>/dev/null || true

# Environment defaults for HF Spaces
export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
export VLLM_PORT="${VLLM_PORT:-8002}"
export API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
export FRONTEND_URL="${FRONTEND_URL:-http://localhost:7860}"
export MODEL_STORAGE_PATH="${MODEL_STORAGE_PATH:-./models}"
export DATA_STORAGE_PATH="${DATA_STORAGE_PATH:-./data}"

mkdir -p "$MODEL_STORAGE_PATH" "$DATA_STORAGE_PATH"

echo "🧠 Starting vLLM on ${VLLM_HOST}:${VLLM_PORT}..."
python -m vllm.entrypoints.api_server \
  --model openai/gpt-oss-20b \
  --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --gpu-memory-utilization 0.7 \
  --max-model-len 2048 \
  --enable-log-requests=false &

echo "⚡ Starting FastAPI on :8000..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level info &

echo "🖥️  Starting Streamlit on :7860..."
python -m streamlit run frontend/app.py \
  --server.address 0.0.0.0 \
  --server.port 7860 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false


