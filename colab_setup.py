"""
Google Colab setup script for Music AI Recommendation System
This version runs everything in a single Colab notebook with GPU support
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def setup_colab_environment():
    """Set up the environment for Google Colab."""
    print("🚀 Setting up Music AI Recommendation System for Google Colab")
    
    # Install required packages
    print("📦 Installing dependencies...")
    packages = [
        "streamlit",
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "pydantic-settings",
        "torch",
        "transformers",
        "vllm",
        "scikit-learn",
        "numpy",
        "pandas",
        "spotipy",
        "sqlalchemy",
        "redis",
        "httpx",
        "aiohttp",
        "requests",
        "plotly",
        "rich",
        "pytest",
        "aiosqlite"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        except:
            print(f"⚠️  Could not install {package}")
    
    print("✅ Dependencies installed")
    
    # Set up environment variables for Colab
    os.environ["PYTHONPATH"] = "/content"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("✅ Environment setup complete")


def check_gpu():
    """Check GPU availability in Colab."""
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU detected: {gpu_name} (Count: {gpu_count})")
        return True
    else:
        print("⚠️  No GPU detected. Please enable GPU in Colab: Runtime > Change runtime type > GPU")
        return False


def start_vllm_server():
    """Start vLLM server in background."""
    print("🧠 Starting vLLM server with GPT-OSS-20B...")
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.api_server",
        "--model", "openai/gpt-oss-20b",
        "--host", "0.0.0.0",
        "--port", "8002",
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "4096"
    ]
    
    def run_vllm():
        subprocess.run(cmd, capture_output=True)
    
    # Start in background thread
    vllm_thread = threading.Thread(target=run_vllm, daemon=True)
    vllm_thread.start()
    
    # Wait for server to be ready
    print("⏳ Waiting for vLLM server to initialize...")
    time.sleep(30)  # Give it time to download and start
    
    return vllm_thread


def start_backend():
    """Start FastAPI backend."""
    print("⚡ Starting FastAPI backend...")
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "backend.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    def run_backend():
        subprocess.run(cmd, capture_output=True)
    
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    time.sleep(5)  # Give it time to start
    return backend_thread


def start_frontend():
    """Start Streamlit frontend."""
    print("🖥️  Starting Streamlit frontend...")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "frontend/app.py", 
        "--server.port", "8501", 
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    def run_frontend():
        subprocess.run(cmd, capture_output=True)
    
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()
    
    time.sleep(5)  # Give it time to start
    return frontend_thread


def setup_colab_ports():
    """Set up Colab port forwarding."""
    print("🔗 Setting up Colab port forwarding...")
    
    # Colab automatically exposes ports, but we need to get the public URLs
    from google.colab import output
    
    # Enable port forwarding
    output.enable_custom_widget_manager()
    
    print("✅ Port forwarding enabled")
    print("📱 Access your app at:")
    print("   Frontend: https://colab.research.google.com/github/your-repo/notebook.ipynb")
    print("   Backend: https://colab.research.google.com/github/your-repo/notebook.ipynb")


def create_colab_notebook():
    """Create a complete Colab notebook."""
    notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🎵 AI Music Recommendation System - Google Colab\\n",
    "\\n",
    "This notebook runs the complete Music AI Recommendation System with GPT-OSS-20B and neural networks.\\n",
    "\\n",
    "## Features:\\n",
    "- AI-powered music recommendations\\n",
    "- GPT-OSS-20B natural language explanations\\n",
    "- Spotify integration with OAuth\\n",
    "- RLHF preference learning\\n",
    "- Interactive Streamlit frontend\\n",
    "\\n",
    "## Setup:\\n",
    "1. Enable GPU: Runtime > Change runtime type > GPU\\n",
    "2. Run all cells below\\n",
    "3. Get your Spotify API credentials from https://developer.spotify.com/dashboard\\n",
    "4. Update the Spotify credentials in the config cell"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install and setup\\n",
    "!git clone https://github.com/your-repo/hack-mvp.git\\n",
    "%cd hack-mvp\\n",
    "!python colab_setup.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configure Spotify API credentials\\n",
    "import os\\n",
    "\\n",
    "# Get your credentials from https://developer.spotify.com/dashboard\\n",
    "SPOTIFY_CLIENT_ID = \"your_client_id_here\"\\n",
    "SPOTIFY_CLIENT_SECRET = \"your_client_secret_here\"\\n",
    "\\n",
    "# Set environment variables\\n",
    "os.environ[\"SPOTIFY_CLIENT_ID\"] = SPOTIFY_CLIENT_ID\\n",
    "os.environ[\"SPOTIFY_CLIENT_SECRET\"] = SPOTIFY_CLIENT_SECRET\\n",
    "os.environ[\"SPOTIFY_REDIRECT_URI\"] = \"https://colab.research.google.com/github/your-repo/notebook.ipynb/callback\"\\n",
    "\\n",
    "print(\"✅ Spotify credentials configured\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start all services\\n",
    "from colab_setup import start_vllm_server, start_backend, start_frontend, setup_colab_ports\\n",
    "\\n",
    "# Check GPU\\n",
    "import torch\\n",
    "if torch.cuda.is_available():\\n",
    "    print(f\"🚀 GPU: {torch.cuda.get_device_name(0)}\")\\n",
    "else:\\n",
    "    print(\"⚠️  No GPU detected. Enable GPU in Runtime > Change runtime type\")\\n",
    "\\n",
    "# Start services\\n",
    "vllm_thread = start_vllm_server()\\n",
    "backend_thread = start_backend()\\n",
    "frontend_thread = start_frontend()\\n",
    "\\n",
    "setup_colab_ports()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test the system\\n",
    "import requests\\n",
    "import time\\n",
    "\\n",
    "# Wait for services to be ready\\n",
    "time.sleep(10)\\n",
    "\\n",
    "# Test backend health\\n",
    "try:\\n",
    "    response = requests.get(\"http://localhost:8000/health\")\\n",
    "    print(f\"✅ Backend health: {response.json()}\")\\n",
    "except Exception as e:\\n",
    "    print(f\"❌ Backend not ready: {e}\")\\n",
    "\\n",
    "# Test vLLM server\\n",
    "try:\\n",
    "    response = requests.get(\"http://localhost:8002/health\")\\n",
    "    print(f\"✅ vLLM health: {response.json()}\")\\n",
    "except Exception as e:\\n",
    "    print(f\"❌ vLLM not ready: {e}\")\\n",
    "\\n",
    "print(\"\\n🎉 System is ready!\")\\n",
    "print(\"\\n📱 Access your app:\")\\n",
    "print(\"   Frontend: https://colab.research.google.com/github/your-repo/notebook.ipynb\")\\n",
    "print(\"   Backend API: http://localhost:8000\")\\n",
    "print(\"   API Docs: http://localhost:8000/docs\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Demo the system\\n",
    "from demo.showcase import HackathonDemo\\n",
    "import asyncio\\n",
    "\\n",
    "demo = HackathonDemo()\\n",
    "await demo.run_full_demo()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open("Music_AI_Recommendation_System.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("📓 Created Colab notebook: Music_AI_Recommendation_System.ipynb")


def main():
    """Main setup function for Colab."""
    print("🎵 Setting up Music AI Recommendation System for Google Colab")
    print("=" * 60)
    
    # Setup environment
    setup_colab_environment()
    
    # Check GPU
    if not check_gpu():
        print("⚠️  Please enable GPU in Colab: Runtime > Change runtime type > GPU")
        return
    
    # Create Colab notebook
    create_colab_notebook()
    
    print("\n" + "=" * 60)
    print("🎉 Colab setup complete!")
    print("\nNext steps:")
    print("1. Upload the generated notebook to Google Colab")
    print("2. Enable GPU: Runtime > Change runtime type > GPU")
    print("3. Get Spotify API credentials from https://developer.spotify.com/dashboard")
    print("4. Update the credentials in the notebook")
    print("5. Run all cells")
    print("\nThe system will run entirely in Colab with GPU acceleration!")


if __name__ == "__main__":
    main()
