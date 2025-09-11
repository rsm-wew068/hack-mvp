# Hugging Face Spaces compatible Dockerfile (GPU)
# Use HF CUDA runtime base (pullable in Spaces builders)
FROM huggingface/cuda:12.1-runtime-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential curl git procps \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first for caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app

# HF requires port 7860 exposed for the main app (Streamlit)
EXPOSE 7860 8000 8002

# Launcher to start vLLM, FastAPI, then Streamlit on 7860
RUN chmod +x /app/scripts/hf_start.sh
CMD ["/bin/bash", "/app/scripts/hf_start.sh"]
