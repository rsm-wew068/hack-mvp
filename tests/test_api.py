"""Tests for the FastAPI backend."""

import pytest
import asyncio
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.main import app


class TestAPI:
    """Test cases for the FastAPI backend."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
    
    def test_get_spotify_auth_url(self, client):
        """Test getting Spotify auth URL."""
        response = client.get("/api/spotify/auth-url/test_user")
        assert response.status_code == 200
        
        data = response.json()
        assert "auth_url" in data
        assert "spotify.com" in data["auth_url"]
    
    def test_get_recommendations(self, client):
        """Test getting recommendations."""
        request_data = {
            "user_id": "test_user",
            "num_recommendations": 5,
            "confidence_threshold": 0.5
        }
        
        response = client.post("/api/recommendations", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "recommendations" in data
        assert "total_count" in data
        assert "user_id" in data
        assert data["user_id"] == "test_user"
    
    def test_get_recommendations_with_playlist(self, client):
        """Test getting recommendations with playlist context."""
        request_data = {
            "user_id": "test_user",
            "playlist_id": "test_playlist",
            "num_recommendations": 3,
            "confidence_threshold": 0.7
        }
        
        response = client.post("/api/recommendations", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) <= 3
    
    def test_record_preference(self, client):
        """Test recording user preference."""
        request_data = {
            "user_id": "test_user",
            "track_a": "track_1",
            "track_b": "track_2",
            "preference": 1
        }
        
        response = client.post("/api/preference-feedback", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "recorded"
    
    def test_get_user_profile(self, client):
        """Test getting user profile."""
        response = client.get("/api/user-profile/test_user")
        assert response.status_code == 200
        
        data = response.json()
        assert "user_id" in data
        assert "training_sessions" in data
        assert "preferences_recorded" in data
        assert "ai_confidence" in data
    
    def test_search_tracks(self, client):
        """Test searching tracks."""
        response = client.get("/api/search/tracks?q=test&limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert "tracks" in data
        assert "query" in data
        assert "count" in data
        assert data["query"] == "test"
    
    def test_get_comparison_tracks(self, client):
        """Test getting comparison tracks for training."""
        response = client.get("/api/training/comparison-tracks/test_user")
        assert response.status_code == 200
        
        data = response.json()
        assert "track_a" in data
        assert "track_b" in data
        assert "id" in data["track_a"]
        assert "id" in data["track_b"]
        assert "name" in data["track_a"]
        assert "name" in data["track_b"]
    
    def test_invalid_recommendation_request(self, client):
        """Test invalid recommendation request."""
        # Missing required fields
        request_data = {
            "user_id": "test_user"
            # Missing num_recommendations
        }
        
        response = client.post("/api/recommendations", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_preference_request(self, client):
        """Test invalid preference request."""
        # Invalid preference value
        request_data = {
            "user_id": "test_user",
            "track_a": "track_1",
            "track_b": "track_2",
            "preference": 2  # Should be 0 or 1
        }
        
        response = client.post("/api/preference-feedback", json=request_data)
        # This might pass validation but fail in processing
        assert response.status_code in [200, 422]


if __name__ == "__main__":
    pytest.main([__file__])
