"""FastAPI backend for the Music AI Recommendation System."""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from recommender.engine import RecommendationEngine
from integrations.spotify_client import SpotifyClient
from models.vllm_server import get_llm_server
from config.settings import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Music AI Recommendation API",
    description="AI-powered music recommendations using GPT-OSS-20B and neural networks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
recommendation_engine: Optional[RecommendationEngine] = None
spotify_client: Optional[SpotifyClient] = None


# Pydantic models for API requests/responses
class RecommendationRequest(BaseModel):
    user_id: str
    playlist_id: Optional[str] = None
    num_recommendations: int = 10
    confidence_threshold: float = 0.5


class PreferenceFeedback(BaseModel):
    user_id: str
    track_a: str
    track_b: str
    preference: int  # 1 for track_a, 0 for track_b


class OAuthCallback(BaseModel):
    user_id: str
    code: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global recommendation_engine, spotify_client
    
    logger.info("Starting Music AI Recommendation System...")
    
    try:
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine()
        logger.info("Recommendation engine initialized")
        
        # Initialize Spotify client
        spotify_client = SpotifyClient()
        logger.info("Spotify client initialized")
        
        # Initialize LLM server
        llm_server = await get_llm_server()
        logger.info("LLM server initialized")
        
        logger.info("All services started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Music AI Recommendation System...")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check LLM server health
        llm_server = await get_llm_server()
        llm_health = await llm_server.health_check()
        
        return {
            "status": "healthy",
            "services": {
                "recommendation_engine": recommendation_engine is not None,
                "spotify_client": spotify_client is not None,
                "llm_server": llm_health.get("healthy", False)
            },
            "llm_status": llm_health.get("status", "unknown")
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Spotify OAuth endpoints
@app.get("/api/spotify/auth-url/{user_id}")
async def get_spotify_auth_url(user_id: str):
    """Get Spotify OAuth URL for user authentication."""
    try:
        if not spotify_client:
            raise HTTPException(status_code=500, detail="Spotify client not initialized")
        
        auth_url = spotify_client.get_oauth_url(user_id)
        return {"auth_url": auth_url}
    except Exception as e:
        logger.error(f"Failed to get auth URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/spotify/callback")
async def handle_spotify_callback(callback: OAuthCallback):
    """Handle Spotify OAuth callback."""
    try:
        if not spotify_client:
            raise HTTPException(status_code=500, detail="Spotify client not initialized")
        
        token_info = await spotify_client.handle_oauth_callback(
            callback.user_id, callback.code
        )
        
        return {
            "status": "success",
            "user_id": callback.user_id,
            "access_token": token_info.get("access_token"),
            "expires_in": token_info.get("expires_in")
        }
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/spotify/user/{user_id}")
async def get_user_info(user_id: str):
    """Get user information from Spotify."""
    try:
        if not spotify_client:
            raise HTTPException(status_code=500, detail="Spotify client not initialized")
        
        user_info = await spotify_client.get_user_info(user_id)
        return user_info
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/spotify/playlists/{user_id}")
async def get_user_playlists(user_id: str):
    """Get user's Spotify playlists."""
    try:
        if not spotify_client:
            raise HTTPException(status_code=500, detail="Spotify client not initialized")
        
        playlists = await spotify_client.get_user_playlists(user_id)
        return {"playlists": playlists}
    except Exception as e:
        logger.error(f"Failed to get playlists: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Recommendation endpoints
@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Get AI-powered music recommendations."""
    try:
        if not recommendation_engine:
            raise HTTPException(status_code=500, detail="Recommendation engine not initialized")
        
        recommendations = await recommendation_engine.get_recommendations(
            user_id=request.user_id,
            playlist_id=request.playlist_id,
            num_recommendations=request.num_recommendations
        )
        
        # Filter by confidence threshold
        filtered_recs = [
            rec for rec in recommendations 
            if rec.get('confidence', 0) >= request.confidence_threshold
        ]
        
        return {
            "recommendations": filtered_recs,
            "total_count": len(filtered_recs),
            "user_id": request.user_id
        }
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/preference-feedback")
async def record_preference(feedback: PreferenceFeedback, background_tasks: BackgroundTasks):
    """Record user preference for RLHF training."""
    try:
        if not recommendation_engine:
            raise HTTPException(status_code=500, detail="Recommendation engine not initialized")
        
        # Record preference in background
        background_tasks.add_task(
            recommendation_engine.record_preference,
            feedback.user_id,
            feedback.track_a,
            feedback.track_b,
            feedback.preference
        )
        
        return {"status": "recorded", "message": "Preference recorded successfully"}
    except Exception as e:
        logger.error(f"Failed to record preference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user-profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get learned user preferences and training history."""
    try:
        if not recommendation_engine:
            raise HTTPException(status_code=500, detail="Recommendation engine not initialized")
        
        profile = await recommendation_engine.get_user_profile(user_id)
        return profile
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Track search endpoint
@app.get("/api/search/tracks")
async def search_tracks(q: str, limit: int = 20):
    """Search for tracks using Spotify API."""
    try:
        if not spotify_client:
            raise HTTPException(status_code=500, detail="Spotify client not initialized")
        
        tracks = await spotify_client.search_tracks(q, limit)
        return {"tracks": tracks, "query": q, "count": len(tracks)}
    except Exception as e:
        logger.error(f"Failed to search tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Comparison tracks for training
@app.get("/api/training/comparison-tracks/{user_id}")
async def get_comparison_tracks(user_id: str):
    """Get two random tracks for preference comparison."""
    try:
        if not spotify_client:
            raise HTTPException(status_code=500, detail="Spotify client not initialized")
        
        # Get some popular tracks for comparison
        # In production, this would use more sophisticated selection
        popular_tracks = await spotify_client.search_tracks("year:2023", limit=50)
        
        if len(popular_tracks) < 2:
            raise HTTPException(status_code=404, detail="Not enough tracks found")
        
        import random
        selected_tracks = random.sample(popular_tracks, 2)
        
        # Get detailed track information
        track_a = await spotify_client.get_track_details(selected_tracks[0]['id'])
        track_b = await spotify_client.get_track_details(selected_tracks[1]['id'])
        
        return {
            "track_a": track_a,
            "track_b": track_b
        }
    except Exception as e:
        logger.error(f"Failed to get comparison tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
