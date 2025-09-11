# Multi-stage Dockerfile for Music AI Recommendation System

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Backend service
FROM base as backend

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data logs

# Expose ports
EXPOSE 8000 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Frontend service
FROM base as frontend

# Copy frontend code
COPY frontend/ ./frontend/
COPY config/ ./config/
COPY integrations/ ./integrations/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# Stage 4: vLLM service
FROM base as vllm

# Install vLLM with CUDA support
RUN pip install --no-cache-dir vllm

# Copy model configuration
COPY models/ ./models/

# Expose vLLM port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Default command
CMD ["python", "-m", "vllm.entrypoints.api_server", \
     "--model", "openai-community/gpt-oss-20b", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--gpu-memory-utilization", "0.8", \
     "--max-model-len", "4096"]
