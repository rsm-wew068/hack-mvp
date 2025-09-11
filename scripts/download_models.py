#!/usr/bin/env python3
"""Script to download and setup AI models for the Music AI Recommendation System."""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.neural_networks import MusicEmbeddingNet, BradleyTerryModel
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_gpt_oss_20b():
    """Download GPT-OSS-20B model for vLLM."""
    print("📥 Setting up GPT-OSS-20B model for vLLM...")
    
    model_name = "openai-community/gpt-oss-20b"
    cache_dir = f"{settings.model_storage_path}/gpt-oss-20b"
    
    try:
        # For vLLM, we just need to ensure the model is cached
        # The actual loading will be handled by vLLM server
        from transformers import AutoTokenizer
        
        print(f"   Caching tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print("✅ GPT-OSS-20B ready for vLLM")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not setup GPT-OSS-20B: {e}")
        print("   The model will be downloaded automatically when vLLM starts")
        return False


def setup_neural_network_models():
    """Initialize and save base neural network models."""
    print("🧠 Setting up neural network models...")
    
    try:
        # Audio embedding model
        print("   Creating audio embedding model...")
        audio_model = MusicEmbeddingNet(num_audio_features=13, embedding_dim=128)
        torch.save(audio_model.state_dict(), f'{settings.model_storage_path}/audio_embedding.pth')
        
        # Bradley-Terry preference model
        print("   Creating Bradley-Terry preference model...")
        bt_model = BradleyTerryModel(num_items=100000, embedding_dim=64)
        torch.save(bt_model.state_dict(), f'{settings.model_storage_path}/bradley_terry.pth')
        
        # Collaborative filtering model (placeholder)
        print("   Creating collaborative filtering model...")
        from models.neural_networks import DeepCollaborativeFilter
        cf_model = DeepCollaborativeFilter(num_users=10000, num_items=100000, embedding_dim=128)
        torch.save(cf_model.state_dict(), f'{settings.model_storage_path}/collaborative_filter.pth')
        
        print("✅ Neural network models initialized")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup neural network models: {e}")
        return False


def download_sample_data():
    """Download sample training data."""
    print("📊 Setting up sample data...")
    
    try:
        # Create sample audio features data
        import numpy as np
        import json
        
        # Generate sample audio features for demo
        sample_features = {
            "4iV5W9uYEdYUVa79Axb7Rh": {  # Never Gonna Give You Up
                "danceability": 0.778,
                "energy": 0.694,
                "valence": 0.931,
                "tempo": 113.928,
                "acousticness": 0.000234,
                "instrumentalness": 0.0,
                "speechiness": 0.0457,
                "liveness": 0.0894,
                "loudness": -5.883
            },
            "3n3Ppam7vgaVa1iaRUpq9s": {  # Sample track
                "danceability": 0.65,
                "energy": 0.8,
                "valence": 0.7,
                "tempo": 120.0,
                "acousticness": 0.1,
                "instrumentalness": 0.0,
                "speechiness": 0.05,
                "liveness": 0.1,
                "loudness": -6.0
            }
        }
        
        # Save sample data
        with open(f"{settings.data_storage_path}/sample_audio_features.json", "w") as f:
            json.dump(sample_features, f, indent=2)
        
        print("✅ Sample data created")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def check_gpu_availability():
    """Check if GPU is available."""
    print("🔍 Checking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name} (Count: {gpu_count})")
        return True
    else:
        print("⚠️  No GPU detected. Models will run on CPU (slower)")
        return False


def main():
    """Main setup function."""
    print("🎵 Music AI Recommendation System - Model Setup")
    print("=" * 50)
    
    # Create directories
    os.makedirs(settings.model_storage_path, exist_ok=True)
    os.makedirs(settings.data_storage_path, exist_ok=True)
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Setup models
    success_count = 0
    
    if gpu_available:
        if download_gpt_oss_20b():
            success_count += 1
    else:
        print("⚠️  Skipping GPT-OSS-20B setup (no GPU)")
    
    if setup_neural_network_models():
        success_count += 1
    
    if download_sample_data():
        success_count += 1
    
    print("\n" + "=" * 50)
    if success_count >= 2:  # At least neural networks and sample data
        print("🎉 Model setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy env.example to .env and add your Spotify API credentials")
        print("2. Run: python scripts/start_app.sh")
        print("3. Open http://localhost:8501 in your browser")
        return 0
    else:
        print("❌ Model setup failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
