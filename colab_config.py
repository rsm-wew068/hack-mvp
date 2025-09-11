"""
Colab-specific configuration for Music AI Recommendation System
Handles Colab's unique requirements and limitations
"""

import os
import sys
from pathlib import Path

# Colab-specific settings
COLAB_MODE = True
COLAB_BASE_URL = "https://colab.research.google.com"

class ColabSettings:
    """Colab-specific configuration settings."""
    
    def __init__(self):
        self.is_colab = self._detect_colab()
        self.setup_colab_environment()
    
    def _detect_colab(self):
        """Detect if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def setup_colab_environment(self):
        """Set up environment variables for Colab."""
        if self.is_colab:
            # Colab-specific paths
            os.environ["PYTHONPATH"] = "/content"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            # Colab port configuration
            os.environ["VLLM_HOST"] = "0.0.0.0"
            os.environ["VLLM_PORT"] = "8002"
            os.environ["API_BASE_URL"] = "http://localhost:8000"
            os.environ["FRONTEND_URL"] = "http://localhost:8501"
            
            # Colab-specific Spotify redirect URI
            if "SPOTIFY_REDIRECT_URI" not in os.environ:
                os.environ["SPOTIFY_REDIRECT_URI"] = f"{COLAB_BASE_URL}/callback"
            
            # Database configuration for Colab
            os.environ["DATABASE_URL"] = "sqlite:///./data/music_app.db"
            os.environ["REDIS_URL"] = "redis://localhost:6379"
            
            # Model storage
            os.environ["MODEL_STORAGE_PATH"] = "./models"
            os.environ["DATA_STORAGE_PATH"] = "./data"
            
            print("✅ Colab environment configured")
    
    def get_colab_urls(self):
        """Get Colab-specific URLs for the application."""
        if not self.is_colab:
            return {
                "frontend": "http://localhost:8501",
                "backend": "http://localhost:8000",
                "api_docs": "http://localhost:8000/docs",
                "vllm": "http://localhost:8002"
            }
        
        # In Colab, we need to use the public URLs
        return {
            "frontend": f"{COLAB_BASE_URL}/notebook.ipynb",
            "backend": "http://localhost:8000",
            "api_docs": "http://localhost:8000/docs",
            "vllm": "http://localhost:8002"
        }
    
    def setup_colab_ports(self):
        """Set up port forwarding for Colab."""
        if not self.is_colab:
            return
        
        try:
            from google.colab import output
            output.enable_custom_widget_manager()
            print("✅ Colab port forwarding enabled")
        except ImportError:
            print("⚠️  Could not enable Colab port forwarding")
    
    def get_spotify_redirect_uri(self):
        """Get the correct Spotify redirect URI for Colab."""
        if self.is_colab:
            return f"{COLAB_BASE_URL}/callback"
        else:
            return "http://localhost:8501/callback"


# Global Colab settings instance
colab_settings = ColabSettings()


def get_colab_config():
    """Get Colab configuration."""
    return colab_settings


def setup_colab_spotify_oauth():
    """Set up Spotify OAuth for Colab."""
    if not colab_settings.is_colab:
        return
    
    print("🔐 Setting up Spotify OAuth for Colab...")
    print("📝 Note: You'll need to update your Spotify app settings:")
    print(f"   Redirect URI: {colab_settings.get_spotify_redirect_uri()}")
    print("   Go to: https://developer.spotify.com/dashboard")
    print("   Edit your app > Settings > Redirect URIs")


def create_colab_startup_script():
    """Create a startup script for Colab."""
    script_content = '''
# Colab Startup Script for Music AI Recommendation System

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

def start_services():
    """Start all services in Colab."""
    print("🚀 Starting Music AI Recommendation System in Colab...")
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected. Enable GPU in Runtime > Change runtime type")
        return
    
    # Start vLLM server
    print("🧠 Starting vLLM server...")
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.api_server",
        "--model", "openai/gpt-oss-20b",
        "--host", "0.0.0.0",
        "--port", "8002",
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "4096"
    ]
    
    def run_vllm():
        subprocess.run(vllm_cmd, capture_output=True)
    
    vllm_thread = threading.Thread(target=run_vllm, daemon=True)
    vllm_thread.start()
    
    # Start backend
    print("⚡ Starting FastAPI backend...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    def run_backend():
        subprocess.run(backend_cmd, capture_output=True)
    
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend
    print("🖥️  Starting Streamlit frontend...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run",
        "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    def run_frontend():
        subprocess.run(frontend_cmd, capture_output=True)
    
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    frontend_thread.start()
    
    # Wait for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(30)
    
    # Test services
    import requests
    
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ Backend: {response.status_code}")
    except:
        print("❌ Backend not ready")
    
    try:
        response = requests.get("http://localhost:8002/health")
        print(f"✅ vLLM: {response.status_code}")
    except:
        print("❌ vLLM not ready")
    
    print("\\n🎉 Services started!")
    print("📱 Access your app at the Colab notebook URL")
    
    return vllm_thread, backend_thread, frontend_thread

# Run the startup
if __name__ == "__main__":
    start_services()
'''
    
    with open("colab_startup.py", "w") as f:
        f.write(script_content)
    
    print("📝 Created colab_startup.py")


if __name__ == "__main__":
    # Test Colab configuration
    config = get_colab_config()
    print(f"Colab mode: {config.is_colab}")
    print(f"URLs: {config.get_colab_urls()}")
    
    if config.is_colab:
        setup_colab_spotify_oauth()
        create_colab_startup_script()
