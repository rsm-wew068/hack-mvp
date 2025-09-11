"""Configuration settings for the Music AI Recommendation System."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Spotify API Configuration
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str = "http://localhost:8501/callback"
    
    # AI Models Configuration
    vllm_model_path: str = "openai-community/gpt-oss-20b"
    vllm_host: str = "localhost"
    vllm_port: int = 8002
    vllm_gpu_memory_utilization: float = 0.8
    vllm_max_model_len: int = 4096
    
    # Database Configuration
    database_url: str = "sqlite:///./music_app.db"
    redis_url: str = "redis://localhost:6379"
    
    # API Configuration
    api_base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:8501"
    
    # Development Settings
    debug: bool = True
    log_level: str = "INFO"
    
    # Model Storage
    model_storage_path: str = "./models"
    data_storage_path: str = "./data"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
