"""Recommendation engine combining multiple AI approaches."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from models.neural_networks import (
    MusicEmbeddingNet, DeepCollaborativeFilter, BradleyTerryModel, RLHFTrainer
)
from models.vllm_server import get_llm_server
from integrations.spotify_client import SpotifyClient
from config.settings import settings

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Main recommendation engine combining multiple AI approaches."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.spotify_client = SpotifyClient()
        self.models = self._load_models()
        self.rlhf_trainer = RLHFTrainer(self.models['bradley_terry'])
        self.scaler = StandardScaler()
        self._user_embeddings = {}  # Cache for user embeddings
        self._track_embeddings = {}  # Cache for track embeddings
        
    def _load_models(self) -> Dict[str, Any]:
        """Load or initialize neural network models."""
        try:
            # Instantiate models and load state_dicts if present
            audio_model = MusicEmbeddingNet(num_audio_features=13, embedding_dim=128)
            cf_model = DeepCollaborativeFilter(num_users=10000, num_items=100000, embedding_dim=128)
            bt_model = BradleyTerryModel(num_items=100000, embedding_dim=64)

            audio_sd = torch.load(
                f"{settings.model_storage_path}/audio_embedding.pth", map_location='cpu'
            )
            cf_sd = torch.load(
                f"{settings.model_storage_path}/collaborative_filter.pth", map_location='cpu'
            )
            bt_sd = torch.load(
                f"{settings.model_storage_path}/bradley_terry.pth", map_location='cpu'
            )

            if isinstance(audio_sd, dict):
                audio_model.load_state_dict(audio_sd)
            if isinstance(cf_sd, dict):
                cf_model.load_state_dict(cf_sd)
            if isinstance(bt_sd, dict):
                bt_model.load_state_dict(bt_sd)

            audio_model.eval()
            cf_model.eval()
            bt_model.eval()

            models = {
                'audio_embedding': audio_model,
                'collaborative': cf_model,
                'bradley_terry': bt_model,
            }
            logger.info("Loaded pre-trained model weights")
        except FileNotFoundError:
            # Initialize new models
            logger.info("Initializing new models")
            models = {
                'audio_embedding': MusicEmbeddingNet(num_audio_features=13, embedding_dim=128),
                'collaborative': DeepCollaborativeFilter(num_users=10000, num_items=100000, embedding_dim=128),
                'bradley_terry': BradleyTerryModel(num_items=100000, embedding_dim=64)
            }
            
            # Save initial models
            import os
            os.makedirs(settings.model_storage_path, exist_ok=True)
            for name, model in models.items():
                torch.save(model.state_dict(), f"{settings.model_storage_path}/{name}.pth")
        
        return models
    
    async def get_recommendations(self, user_id: str, playlist_id: Optional[str] = None, 
                                num_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Get AI-powered music recommendations for a user.
        
        Args:
            user_id: User identifier
            playlist_id: Optional playlist ID for context-based recommendations
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries with explanations
        """
        try:
            # Get collaborative filtering recommendations
            collab_recs = await self._collaborative_filtering(user_id, num_recommendations * 2)
            
            # Get audio similarity recommendations if playlist provided
            audio_recs = []
            if playlist_id:
                playlist_tracks = await self.spotify_client.get_playlist_tracks(playlist_id, user_id)
                audio_recs = await self._audio_similarity_recommendations(
                    playlist_tracks, num_recommendations * 2
                )
            
            # Combine and rank using RLHF preferences
            combined_recs = self._combine_recommendations(collab_recs, audio_recs, user_id)
            
            # Get top recommendations with confidence scores
            final_recs = combined_recs[:num_recommendations]
            
            # Enrich with Spotify metadata and AI explanations
            llm_server = await get_llm_server()
            enriched_recs = []
            
            for rec in final_recs:
                # Get detailed track info
                track_details = await self.spotify_client.get_track_details(rec['track_id'])
                audio_features = await self.spotify_client.get_audio_features(rec['track_id'])
                
                # Generate AI explanation
                user_preferences = await self._get_user_preferences(user_id)
                explanation = await llm_server.explain_recommendation(
                    track_info={**track_details, 'audio_features': audio_features},
                    user_preferences=user_preferences,
                    confidence_score=rec['confidence']
                )
                
                enriched_rec = {
                    **rec,
                    **track_details,
                    'audio_features': audio_features,
                    'explanation': explanation,
                    'spotify_url': track_details.get('external_urls', {}).get('spotify', '')
                }
                enriched_recs.append(enriched_rec)
            
            return enriched_recs
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []
    
    async def _collaborative_filtering(self, user_id: str, num_recs: int) -> List[Dict[str, Any]]:
        """Generate collaborative filtering recommendations."""
        try:
            # For demo purposes, return mock recommendations
            # In production, this would use the trained collaborative filtering model
            mock_tracks = [
                {'track_id': '4iV5W9uYEdYUVa79Axb7Rh', 'confidence': 0.85},  # Never Gonna Give You Up
                {'track_id': '3n3Ppam7vgaVa1iaRUpq9s', 'confidence': 0.82},  # Some other track
                {'track_id': '0VjIjW4X0L0j4K0j4K0j4K', 'confidence': 0.78},  # Mock track
            ]
            return mock_tracks[:num_recs]
        except Exception as e:
            logger.error(f"Collaborative filtering failed: {e}")
            return []
    
    async def _audio_similarity_recommendations(self, playlist_tracks: List[Dict[str, Any]], 
                                             num_recs: int) -> List[Dict[str, Any]]:
        """Generate recommendations based on audio similarity."""
        try:
            if not playlist_tracks:
                return []
            
            # Get audio features for playlist tracks
            playlist_features = []
            for track in playlist_tracks[:5]:  # Use first 5 tracks for efficiency
                features = await self.spotify_client.get_audio_features(track['id'])
                if features:
                    feature_vector = [
                        features.get('danceability', 0.5),
                        features.get('energy', 0.5),
                        features.get('valence', 0.5),
                        features.get('acousticness', 0.5),
                        features.get('instrumentalness', 0.5),
                        features.get('speechiness', 0.5),
                        features.get('liveness', 0.5),
                        features.get('tempo', 120) / 200.0,  # Normalize tempo
                        features.get('loudness', -10) / 60.0,  # Normalize loudness
                    ]
                    playlist_features.append(feature_vector)
            
            if not playlist_features:
                return []
            
            # Calculate average features
            avg_features = np.mean(playlist_features, axis=0)
            
            # For demo, return mock similar tracks
            # In production, this would search a database of tracks with similar features
            similar_tracks = [
                {'track_id': '4iV5W9uYEdYUVa79Axb7Rh', 'confidence': 0.88},
                {'track_id': '3n3Ppam7vgaVa1iaRUpq9s', 'confidence': 0.85},
                {'track_id': '0VjIjW4X0L0j4K0j4K0j4K', 'confidence': 0.81},
            ]
            
            return similar_tracks[:num_recs]
            
        except Exception as e:
            logger.error(f"Audio similarity recommendations failed: {e}")
            return []
    
    def _combine_recommendations(self, collab_recs: List[Dict[str, Any]], 
                               audio_recs: List[Dict[str, Any]], 
                               user_id: str) -> List[Dict[str, Any]]:
        """Combine and rank recommendations from different sources."""
        try:
            # Combine all recommendations
            all_recs = {}
            
            # Add collaborative filtering recommendations
            for rec in collab_recs:
                track_id = rec['track_id']
                if track_id not in all_recs:
                    all_recs[track_id] = {'track_id': track_id, 'confidence': 0, 'sources': []}
                all_recs[track_id]['confidence'] += rec['confidence'] * 0.6  # Weight for CF
                all_recs[track_id]['sources'].append('collaborative')
            
            # Add audio similarity recommendations
            for rec in audio_recs:
                track_id = rec['track_id']
                if track_id not in all_recs:
                    all_recs[track_id] = {'track_id': track_id, 'confidence': 0, 'sources': []}
                all_recs[track_id]['confidence'] += rec['confidence'] * 0.4  # Weight for audio
                all_recs[track_id]['sources'].append('audio_similarity')
            
            # Apply RLHF preferences if available
            # This would use the Bradley-Terry model to adjust confidence scores
            
            # Sort by confidence and return
            sorted_recs = sorted(all_recs.values(), key=lambda x: x['confidence'], reverse=True)
            return sorted_recs
            
        except Exception as e:
            logger.error(f"Error combining recommendations: {e}")
            return []
    
    async def record_preference(self, user_id: str, track_a: str, track_b: str, preference: int):
        """
        Record user preference for RLHF training.
        
        Args:
            user_id: User identifier
            track_a: First track ID
            track_b: Second track ID
            preference: 1 if track_a preferred, 0 if track_b preferred
        """
        try:
            # Store preference in database (would be implemented)
            await self._store_preference(user_id, track_a, track_b, preference)
            
            # Update Bradley-Terry model
            # Convert track IDs to indices (would use a proper mapping in production)
            track_a_idx = hash(track_a) % 100000
            track_b_idx = hash(track_b) % 100000
            
            comparison = (track_a_idx, track_b_idx, preference)
            loss = self.rlhf_trainer.update_preferences([comparison])
            
            logger.info(f"Updated RLHF model with preference. Loss: {loss:.4f}")
            
            # Periodically save updated model
            if self._should_save_model():
                torch.save(self.models['bradley_terry'].state_dict(), 
                          f"{settings.model_storage_path}/bradley_terry.pth")
                
        except Exception as e:
            logger.error(f"Error recording preference: {e}")
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get learned user preferences and training history."""
        try:
            # Mock user profile data
            # In production, this would query the database
            profile = {
                'user_id': user_id,
                'training_sessions': 15,
                'preferences_recorded': 42,
                'ai_confidence': 0.78,
                'recommendations_given': 156,
                'learned_preferences': {
                    'energy': 0.72,
                    'danceability': 0.68,
                    'valence': 0.65,
                    'acousticness': 0.31
                },
                'training_history': [
                    {'date': '2024-01-01', 'accuracy': 0.65},
                    {'date': '2024-01-02', 'accuracy': 0.68},
                    {'date': '2024-01-03', 'accuracy': 0.72},
                    {'date': '2024-01-04', 'accuracy': 0.75},
                    {'date': '2024-01-05', 'accuracy': 0.78},
                ]
            }
            return profile
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {}
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user's learned preferences."""
        profile = await self.get_user_profile(user_id)
        return profile.get('learned_preferences', {})
    
    async def _store_preference(self, user_id: str, track_a: str, track_b: str, preference: int):
        """Store preference in database."""
        # Mock implementation - would use proper database in production
        logger.info(f"Stored preference: {user_id} prefers {track_a if preference else track_b}")
    
    def _should_save_model(self) -> bool:
        """Check if model should be saved (e.g., after N updates)."""
        # Mock implementation - would track update count in production
        return True
