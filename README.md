---
title: Music AI Recommender
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: ""
pinned: false
app_port: 7860
---

# 🎵 AI Music Recommendation System

A complete AI-powered music recommendation system that combines neural networks with GPT-OSS-20B for natural language explanations. Built with FastAPI, Streamlit, and PyTorch.

## 🚀 Features

- **AI-Powered Recommendations**: Combines collaborative filtering, audio similarity, and RLHF preference learning
- **GPT-OSS-20B Integration**: Natural language explanations for each recommendation
- **Spotify Integration**: OAuth authentication and playlist analysis
- **Interactive Training**: A/B comparison interface for RLHF preference learning
- **Real-time Learning**: Continuous model updates based on user feedback
- **Modern UI**: Beautiful Streamlit frontend with analytics and insights

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Services   │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (vLLM/PyTorch)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Spotify API     │    │ Database        │    │ Model Storage   │
│ - OAuth         │    │ - User Prefs    │    │ - Checkpoints   │
│ - Playlists     │    │ - Interactions  │    │ - Embeddings    │
│ - Tracks        │    │ - Models        │    │ - Weights       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for vLLM)
- Spotify Developer Account
- Redis (optional, for caching)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hack-mvp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment**
   ```bash
   cp env.example .env
   # Edit .env with your Spotify API credentials
   ```

4. **Download models**
   ```bash
   python scripts/download_models.py
   ```

5. **Start the application**
   ```bash
   ./scripts/start_app.sh
   ```

6. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

- **Frontend**: Streamlit app on port 8501
- **Backend**: FastAPI server on port 8000
- **vLLM**: GPT-OSS-20B server on port 8002
- **Redis**: Caching and task queue on port 6379
- **PostgreSQL**: Production database on port 5432
- **Celery**: Background task processing
- **Flower**: Task monitoring on port 5555

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Spotify API Configuration
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8501/callback

# AI Models Configuration
VLLM_MODEL_PATH=openai/gpt-oss-20b
VLLM_HOST=localhost
VLLM_PORT=8002

# Database Configuration
DATABASE_URL=sqlite:///./music_app.db
REDIS_URL=redis://localhost:6379

# Development Settings
DEBUG=True
LOG_LEVEL=INFO
```

### Spotify API Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Add `http://localhost:8501/callback` to redirect URIs
4. Copy Client ID and Client Secret to your `.env` file

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_recommendations.py
pytest tests/test_api.py
pytest tests/test_spotify_integration.py

# Run with coverage
pytest --cov=. --cov-report=html
```

## 📊 Usage

### 1. Authentication
- Click "Login to Spotify" in the sidebar
- Grant permissions for playlist access
- Your playlists and preferences will be analyzed

### 2. Get Recommendations
- Select a playlist for context-based recommendations
- Adjust confidence threshold and number of recommendations
- View AI explanations for each recommendation

### 3. Train the AI
- Use the Training tab for A/B comparisons
- Choose between two tracks to teach preferences
- The AI learns from your choices using RLHF

### 4. View Profile
- See your musical taste profile
- Track training progress and AI confidence
- View learned preferences and analytics

## 🔬 AI Models

### Neural Networks
- **MusicEmbeddingNet**: Audio feature embeddings
- **DeepCollaborativeFilter**: User-item interaction modeling
- **BradleyTerryModel**: Pairwise preference learning

### LLM Integration
- **GPT-OSS-20B**: Natural language explanations via vLLM
- **Prompt Engineering**: Contextual recommendation reasoning
- **Async Processing**: Non-blocking explanation generation

### RLHF Training
- **Preference Learning**: Bradley-Terry model updates
- **Continuous Learning**: Real-time model improvements
- **Feedback Loop**: User preferences → Model updates → Better recommendations

## 📈 Performance

### Benchmarks
- **Recommendation Latency**: < 2 seconds
- **LLM Explanation**: < 3 seconds
- **Model Training**: Incremental updates
- **Memory Usage**: ~8GB GPU memory for vLLM

### Optimization
- **Caching**: Redis for frequent queries
- **Async Processing**: Non-blocking operations
- **Model Quantization**: FP16 for efficiency
- **Batch Processing**: Efficient preference updates

## 🚀 Production Deployment

### Health Checks
```bash
# Check all services
python scripts/health_check.py

# Monitor with Docker
docker-compose ps
```

### Scaling
- **Horizontal**: Multiple backend instances
- **Vertical**: GPU memory optimization
- **Caching**: Redis cluster for high availability
- **Database**: PostgreSQL with connection pooling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-OSS-20B
- Spotify for the Web API
- vLLM team for efficient LLM serving
- PyTorch and FastAPI communities
