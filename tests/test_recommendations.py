"""Tests for the recommendation engine."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from recommender.engine import RecommendationEngine
from models.neural_networks import BradleyTerryModel, RLHFTrainer


class TestRecommendationEngine:
    """Test cases for the recommendation engine."""
    
    @pytest.fixture
    def mock_llm_server(self):
        """Mock LLM server for testing."""
        mock = AsyncMock()
        mock.explain_recommendation.return_value = "This track matches your love for energetic pop music with great danceability."
        return mock
    
    @pytest.fixture
    def mock_spotify_client(self):
        """Mock Spotify client for testing."""
        mock = AsyncMock()
        mock.get_track_details.return_value = {
            'id': 'test_track',
            'name': 'Test Track',
            'artist': 'Test Artist',
            'album': 'Test Album',
            'album_art': 'https://example.com/art.jpg',
            'spotify_url': 'https://open.spotify.com/track/test'
        }
        mock.get_audio_features.return_value = {
            'danceability': 0.8,
            'energy': 0.7,
            'valence': 0.6,
            'tempo': 120.0,
            'acousticness': 0.1,
            'instrumentalness': 0.0,
            'speechiness': 0.05,
            'liveness': 0.1,
            'loudness': -6.0
        }
        return mock
    
    @pytest.fixture
    def recommendation_engine(self, mock_llm_server, mock_spotify_client):
        """Create recommendation engine with mocked dependencies."""
        with patch('recommender.engine.get_llm_server', return_value=mock_llm_server), \
             patch('recommender.engine.SpotifyClient', return_value=mock_spotify_client):
            engine = RecommendationEngine()
            return engine
    
    @pytest.mark.asyncio
    async def test_get_recommendations_basic(self, recommendation_engine):
        """Test basic recommendation functionality."""
        recs = await recommendation_engine.get_recommendations(
            user_id="test_user",
            num_recommendations=5
        )
        
        assert len(recs) <= 5
        assert all('confidence' in rec for rec in recs)
        assert all('explanation' in rec for rec in recs)
        assert all('name' in rec for rec in recs)
        assert all('artist' in rec for rec in recs)
    
    @pytest.mark.asyncio
    async def test_get_recommendations_with_playlist(self, recommendation_engine):
        """Test recommendations with playlist context."""
        recs = await recommendation_engine.get_recommendations(
            user_id="test_user",
            playlist_id="test_playlist",
            num_recommendations=3
        )
        
        assert len(recs) <= 3
        assert all('confidence' in rec for rec in recs)
    
    @pytest.mark.asyncio
    async def test_record_preference(self, recommendation_engine):
        """Test recording user preferences."""
        # This should not raise an exception
        await recommendation_engine.record_preference(
            user_id="test_user",
            track_a="track_1",
            track_b="track_2",
            preference=1
        )
    
    @pytest.mark.asyncio
    async def test_get_user_profile(self, recommendation_engine):
        """Test getting user profile."""
        profile = await recommendation_engine.get_user_profile("test_user")
        
        assert isinstance(profile, dict)
        assert 'user_id' in profile
        assert 'training_sessions' in profile
        assert 'preferences_recorded' in profile
        assert 'ai_confidence' in profile


class TestRLHFTrainer:
    """Test cases for RLHF trainer."""
    
    def test_bradley_terry_model_initialization(self):
        """Test Bradley-Terry model initialization."""
        model = BradleyTerryModel(num_items=1000, embedding_dim=64)
        assert model.item_strengths.num_embeddings == 1000
        assert model.strength_head.in_features == 64
    
    def test_rlhf_trainer_initialization(self):
        """Test RLHF trainer initialization."""
        bt_model = BradleyTerryModel(num_items=1000, embedding_dim=64)
        trainer = RLHFTrainer(bt_model)
        
        assert trainer.bt_model == bt_model
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
    
    def test_update_preferences(self):
        """Test preference updates."""
        bt_model = BradleyTerryModel(num_items=1000, embedding_dim=64)
        trainer = RLHFTrainer(bt_model, learning_rate=1e-3)
        
        # Mock comparisons
        comparisons = [
            (0, 1, 1),  # Item 0 preferred over item 1
            (2, 3, 0),  # Item 3 preferred over item 2
        ]
        
        # This should not raise an exception
        loss = trainer.update_preferences(comparisons)
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_get_item_strengths(self):
        """Test getting item strengths."""
        bt_model = BradleyTerryModel(num_items=100, embedding_dim=32)
        trainer = RLHFTrainer(bt_model)
        
        strengths = trainer.get_item_strengths()
        assert strengths.shape == (100,)


class TestNeuralNetworks:
    """Test cases for neural network models."""
    
    def test_music_embedding_net(self):
        """Test music embedding network."""
        from models.neural_networks import MusicEmbeddingNet
        import torch
        
        model = MusicEmbeddingNet(num_audio_features=13, embedding_dim=128)
        
        # Test forward pass
        batch_size = 5
        audio_features = torch.randn(batch_size, 13)
        embeddings = model(audio_features)
        
        assert embeddings.shape == (batch_size, 128)
        # Check if embeddings are normalized
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-6)
    
    def test_deep_collaborative_filter(self):
        """Test deep collaborative filtering model."""
        from models.neural_networks import DeepCollaborativeFilter
        import torch
        
        model = DeepCollaborativeFilter(num_users=1000, num_items=5000, embedding_dim=128)
        
        # Test forward pass
        batch_size = 10
        user_ids = torch.randint(0, 1000, (batch_size,))
        item_ids = torch.randint(0, 5000, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        
        assert predictions.shape == (batch_size,)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)  # Sigmoid output
    
    def test_bradley_terry_model(self):
        """Test Bradley-Terry model."""
        import torch
        
        model = BradleyTerryModel(num_items=1000, embedding_dim=64)
        
        # Test forward pass
        item_a = torch.tensor([0, 1, 2])
        item_b = torch.tensor([3, 4, 5])
        
        predictions = model(item_a, item_b)
        
        assert predictions.shape == (3,)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)  # Sigmoid output


if __name__ == "__main__":
    pytest.main([__file__])
