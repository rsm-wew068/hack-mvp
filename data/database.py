"""Database management for the Music AI Recommendation System."""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import aiosqlite
from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager for SQLite operations."""
    
    def __init__(self, db_path: str = None):
        """Initialize database manager."""
        self.db_path = db_path or settings.database_url.replace("sqlite:///", "")
        self._init_database()
    
    def _init_database(self):
        """Initialize database with schema."""
        try:
            with open("data/schema.sql", "r") as f:
                schema_sql = f.read()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema_sql)
                conn.commit()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def create_user(self, user_id: str, spotify_id: str = None, 
                         display_name: str = None) -> bool:
        """Create a new user."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO users (id, spotify_id, display_name) VALUES (?, ?, ?)",
                    (user_id, spotify_id, display_name)
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to create user {user_id}: {e}")
            return False
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT * FROM users WHERE id = ?", (user_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
            return None
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    async def store_track(self, track_data: Dict[str, Any]) -> bool:
        """Store track information."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT OR REPLACE INTO tracks 
                       (id, name, artist, album, duration_ms, popularity, explicit, audio_features)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        track_data['id'],
                        track_data['name'],
                        track_data['artist'],
                        track_data.get('album'),
                        track_data.get('duration_ms'),
                        track_data.get('popularity'),
                        track_data.get('explicit', False),
                        json.dumps(track_data.get('audio_features', {}))
                    )
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store track {track_data.get('id')}: {e}")
            return False
    
    async def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get track information."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT * FROM tracks WHERE id = ?", (track_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        track_data = dict(zip(columns, row))
                        # Parse JSON fields
                        if track_data.get('audio_features'):
                            track_data['audio_features'] = json.loads(track_data['audio_features'])
                        return track_data
            return None
        except Exception as e:
            logger.error(f"Failed to get track {track_id}: {e}")
            return None
    
    async def create_playlist(self, playlist_data: Dict[str, Any]) -> bool:
        """Create a playlist."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT OR REPLACE INTO playlists 
                       (id, user_id, name, description, tracks_count, public)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        playlist_data['id'],
                        playlist_data['user_id'],
                        playlist_data['name'],
                        playlist_data.get('description'),
                        playlist_data.get('tracks_count', 0),
                        playlist_data.get('public', False)
                    )
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to create playlist {playlist_data.get('id')}: {e}")
            return False
    
    async def add_track_to_playlist(self, playlist_id: str, track_id: str, position: int = None) -> bool:
        """Add track to playlist."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO playlist_tracks (playlist_id, track_id, position) VALUES (?, ?, ?)",
                    (playlist_id, track_id, position)
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add track {track_id} to playlist {playlist_id}: {e}")
            return False
    
    async def store_preference(self, user_id: str, track_a_id: str, track_b_id: str, 
                             preference: int, confidence: float = 1.0) -> bool:
        """Store user preference."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO user_preferences 
                       (user_id, track_a_id, track_b_id, preference, confidence)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, track_a_id, track_b_id, preference, confidence)
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store preference: {e}")
            return False
    
    async def get_recent_preferences(self, user_id: str = None, days: int = 7) -> List[Tuple]:
        """Get recent preferences for training."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                if user_id:
                    query = """SELECT track_a_id, track_b_id, preference 
                              FROM user_preferences 
                              WHERE user_id = ? AND created_at >= datetime('now', '-{} days')""".format(days)
                    params = (user_id,)
                else:
                    query = """SELECT track_a_id, track_b_id, preference 
                              FROM user_preferences 
                              WHERE created_at >= datetime('now', '-{} days')""".format(days)
                    params = ()
                
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return rows
        except Exception as e:
            logger.error(f"Failed to get recent preferences: {e}")
            return []
    
    async def store_recommendation(self, user_id: str, track_id: str, 
                                 confidence_score: float, explanation: str = None,
                                 playlist_id: str = None, source: str = "hybrid") -> bool:
        """Store recommendation."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO recommendations 
                       (user_id, track_id, playlist_id, confidence_score, explanation, source)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, track_id, playlist_id, confidence_score, explanation, source)
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store recommendation: {e}")
            return False
    
    async def get_user_recommendations(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's recommendation history."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """SELECT r.*, t.name, t.artist, t.album 
                       FROM recommendations r
                       JOIN tracks t ON r.track_id = t.id
                       WHERE r.user_id = ?
                       ORDER BY r.created_at DESC
                       LIMIT ?""",
                    (user_id, limit)
                ) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get user recommendations: {e}")
            return []
    
    async def update_recommendation_feedback(self, recommendation_id: int, feedback: int) -> bool:
        """Update recommendation with user feedback."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "UPDATE recommendations SET user_feedback = ? WHERE id = ?",
                    (feedback, recommendation_id)
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update recommendation feedback: {e}")
            return False
    
    async def save_model_checkpoint(self, model_name: str, checkpoint_path: str, 
                                  performance_metrics: Dict[str, Any], 
                                  training_samples: int = 0) -> bool:
        """Save model checkpoint."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO model_checkpoints 
                       (model_name, checkpoint_path, performance_metrics, training_samples)
                       VALUES (?, ?, ?, ?)""",
                    (model_name, checkpoint_path, json.dumps(performance_metrics), training_samples)
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {e}")
            return False
    
    async def get_latest_checkpoint(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get latest model checkpoint."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """SELECT * FROM model_checkpoints 
                       WHERE model_name = ?
                       ORDER BY created_at DESC
                       LIMIT 1""",
                    (model_name,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        checkpoint_data = dict(zip(columns, row))
                        if checkpoint_data.get('performance_metrics'):
                            checkpoint_data['performance_metrics'] = json.loads(
                                checkpoint_data['performance_metrics']
                            )
                        return checkpoint_data
            return None
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            return None
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT * FROM user_stats WHERE id = ?", (user_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        columns = [description[0] for description in cursor.description]
                        return dict(zip(columns, row))
            return {}
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {}
    
    async def cache_audio_features(self, track_id: str, features: Dict[str, Any]) -> bool:
        """Cache audio features for performance."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO audio_features_cache (track_id, features) VALUES (?, ?)",
                    (track_id, json.dumps(features))
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to cache audio features: {e}")
            return False
    
    async def get_cached_audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get cached audio features."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT features FROM audio_features_cache WHERE track_id = ?", (track_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return json.loads(row[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get cached audio features: {e}")
            return None


# Global database manager instance
db_manager = DatabaseManager()
