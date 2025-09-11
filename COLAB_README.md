# 🎵 AI Music Recommendation System - Google Colab Version

Run the complete Music AI Recommendation System with GPT-OSS-20B and neural networks directly in Google Colab Pro!

## 🚀 Quick Start

### 1. **Enable GPU in Colab**
- Go to **Runtime > Change runtime type**
- Select **GPU** (T4 or better recommended)
- Click **Save**

### 2. **Clone and Setup**
```python
# Run this in a Colab cell
!git clone https://github.com/your-repo/hack-mvp.git
%cd hack-mvp
!pip install -r colab_requirements.txt
!python colab_setup.py
```

### 3. **Configure Spotify API**
```python
# Get credentials from https://developer.spotify.com/dashboard
import os

SPOTIFY_CLIENT_ID = "your_client_id_here"
SPOTIFY_CLIENT_SECRET = "your_client_secret_here"

os.environ["SPOTIFY_CLIENT_ID"] = SPOTIFY_CLIENT_ID
os.environ["SPOTIFY_CLIENT_SECRET"] = SPOTIFY_CLIENT_SECRET
os.environ["SPOTIFY_REDIRECT_URI"] = "https://colab.research.google.com/callback"

print("✅ Spotify credentials configured")
```

### 4. **Start All Services**
```python
from colab_config import get_colab_config, setup_colab_spotify_oauth
import subprocess
import threading
import time

# Setup Colab environment
config = get_colab_config()
setup_colab_spotify_oauth()

# Start services
!python colab_startup.py
```

### 5. **Access Your App**
- The Streamlit frontend will be available through Colab's interface
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## 🎯 Features Working in Colab

✅ **AI Recommendations**: Neural networks + collaborative filtering  
✅ **GPT-OSS-20B Explanations**: Natural language reasoning  
✅ **RLHF Training**: Interactive A/B comparisons  
✅ **Spotify Integration**: OAuth and playlist analysis  
✅ **Real-time Learning**: Continuous model updates  
✅ **Beautiful UI**: Streamlit frontend with analytics  

## 🔧 Colab-Specific Configuration

### **Spotify OAuth Setup**
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Add redirect URI: `https://colab.research.google.com/callback`
4. Copy Client ID and Secret to the config cell

### **GPU Requirements**
- **Minimum**: T4 GPU (free tier)
- **Recommended**: V100 or A100 (Colab Pro/Pro+)
- **Memory**: 8GB+ GPU memory for vLLM

### **Port Access**
- Colab automatically handles port forwarding
- Services run on localhost within the Colab environment
- Frontend accessible through Colab's interface

## 📊 Performance in Colab

### **Model Loading**
- GPT-OSS-20B: ~2-3 minutes initial load
- Neural networks: ~30 seconds
- Total startup time: ~5 minutes

### **Recommendation Speed**
- First recommendation: ~10-15 seconds
- Subsequent recommendations: ~2-3 seconds
- LLM explanations: ~3-5 seconds

### **Memory Usage**
- GPU: ~6-8GB for vLLM
- RAM: ~4-6GB total
- Storage: ~2GB for models

## 🐛 Troubleshooting

### **GPU Not Available**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### **Services Not Starting**
```python
# Check if ports are available
!netstat -tlnp | grep -E ":(8000|8002|8501)"
```

### **Spotify OAuth Issues**
- Ensure redirect URI matches exactly
- Check that your Spotify app is not in development mode
- Verify client ID and secret are correct

### **Memory Issues**
```python
# Monitor GPU memory
!nvidia-smi
```

## 🎮 Interactive Demo

```python
# Run the interactive demo
from demo.showcase import HackathonDemo
import asyncio

demo = HackathonDemo()
await demo.run_full_demo()
```

## 📱 Accessing the App

### **Method 1: Colab Interface**
- The Streamlit app will be embedded in the Colab notebook
- Look for the app interface in the output cells

### **Method 2: Direct URLs**
- Frontend: Available through Colab's port forwarding
- Backend: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

## 🔄 Restarting Services

```python
# Restart all services
!pkill -f "streamlit\|uvicorn\|vllm"
!python colab_startup.py
```

## 💡 Tips for Colab

1. **Keep the tab active** to prevent disconnection
2. **Use Colab Pro** for better GPU and longer sessions
3. **Save your work** regularly as Colab sessions can timeout
4. **Monitor GPU usage** with `!nvidia-smi`
5. **Use the demo mode** if Spotify setup is complex

## 🎉 What You Get

- **Complete AI system** running in your browser
- **GPT-OSS-20B** for natural language explanations
- **Neural networks** for personalized recommendations
- **RLHF training** with interactive A/B comparisons
- **Spotify integration** with real playlist analysis
- **Production-ready code** that you can deploy anywhere

## 🚀 Next Steps

1. **Experiment** with different playlists and preferences
2. **Train the AI** using the A/B comparison interface
3. **Explore the code** to understand the implementation
4. **Deploy locally** or to cloud platforms
5. **Extend the system** with additional features

---

**Ready to experience AI-powered music recommendations? Start with the Quick Start guide above!** 🎵
