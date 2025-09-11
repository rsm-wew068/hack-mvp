"""Spotify API client with OAuth integration."""

import logging
from typing import Dict, List, Optional, Any
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from config.settings import settings

logger = logging.getLogger(__name__)


class SpotifyClient:
    """Spotify API client with OAuth and audio features support."""
    
    def __init__(self):
        """Initialize Spotify client with credentials."""
        self.client_credentials_manager = SpotifyClientCredentials(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret
        )
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        self.auth_cache: Dict[str, spotipy.Spotify] = {}  # Store user tokens
    
    def get_oauth_url(self, user_id: str) -> str:
        """Generate Spotify OAuth URL for user authentication."""
        auth_manager = SpotifyOAuth(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            redirect_uri=settings.spotify_redirect_uri,
            scope="playlist-read-private playlist-read-collaborative user-read-private",
            cache_path=f".spotify_cache_{user_id}"
        )
        return auth_manager.get_authorize_url()
    
    async def handle_oauth_callback(self, user_id: str, code: str) -> Dict[str, Any]:
        """Handle OAuth callback and store user token."""
        auth_manager = SpotifyOAuth(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            redirect_uri=settings.spotify_redirect_uri,
            scope="playlist-read-private playlist-read-collaborative user-read-private",
            cache_path=f".spotify_cache_{user_id}"
        )
        
        try:
            token_info = auth_manager.get_access_token(code)
            self.auth_cache[user_id] = spotipy.Spotify(auth=token_info['access_token'])
            return token_info
        except Exception as e:
            logger.error(f"OAuth callback failed for user {user_id}: {e}")
            raise
    
    async def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information from Spotify."""
        if user_id not in self.auth_cache:
            raise Exception("User not authenticated")
        
        sp_user = self.auth_cache[user_id]
        user_info = sp_user.current_user()
        return {
            'id': user_info['id'],
            'display_name': user_info['display_name'],
            'followers': user_info['followers']['total'],
            'country': user_info.get('country', 'Unknown')
        }
    
    async def get_user_playlists(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's personal playlists."""
        if user_id not in self.auth_cache:
            raise Exception("User not authenticated")
        
        sp_user = self.auth_cache[user_id]
        playlists = []
        
        try:
            results = sp_user.current_user_playlists(limit=50)
            while results:
                for playlist in results['items']:
                    playlists.append({
                        'id': playlist['id'],
                        'name': playlist['name'],
                        'description': playlist.get('description', ''),
                        'tracks_count': playlist['tracks']['total'],
                        'public': playlist['public'],
                        'owner': playlist['owner']['display_name']
                    })
                
                if results['next']:
                    results = sp_user.next(results)
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Failed to get playlists for user {user_id}: {e}")
            raise
        
        return playlists
    
    async def get_playlist_tracks(self, playlist_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tracks from a playlist."""
        sp_client = self.auth_cache.get(user_id, self.sp)
        
        tracks = []
        try:
            results = sp_client.playlist_tracks(playlist_id)
            
            while results:
                for item in results['items']:
                    if item['track'] and item['track']['id']:
                        track = item['track']
                        tracks.append({
                            'id': track['id'],
                            'name': track['name'],
                            'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                            'album': track['album']['name'] if track['album'] else 'Unknown',
                            'uri': track['uri'],
                            'duration_ms': track['duration_ms'],
                            'popularity': track.get('popularity', 0),
                            'explicit': track.get('explicit', False)
                        })
                
                if results['next']:
                    results = sp_client.next(results)
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Failed to get tracks for playlist {playlist_id}: {e}")
            raise
        
        return tracks
    
    async def get_audio_features(self, track_id: str) -> Dict[str, float]:
        """Get audio features for a track."""
        try:
            features = self.sp.audio_features(track_id)[0]
            if not features:
                return {}
            
            return {
                'danceability': features['danceability'],
                'energy': features['energy'],
                'valence': features['valence'],
                'tempo': features['tempo'],
                'acousticness': features['acousticness'],
                'instrumentalness': features['instrumentalness'],
                'speechiness': features['speechiness'],
                'liveness': features['liveness'],
                'loudness': features['loudness'],
                'key': features['key'],
                'mode': features['mode'],
                'time_signature': features['time_signature']
            }
        except Exception as e:
            logger.error(f"Failed to get audio features for track {track_id}: {e}")
            return {}
    
    async def get_track_details(self, track_id: str) -> Dict[str, Any]:
        """Get detailed track information."""
        try:
            track = self.sp.track(track_id)
            return {
                'id': track['id'],
                'name': track['name'],
                'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                'album': track['album']['name'] if track['album'] else 'Unknown',
                'album_art': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'duration_ms': track['duration_ms'],
                'popularity': track.get('popularity', 0),
                'explicit': track.get('explicit', False),
                'preview_url': track.get('preview_url'),
                'external_urls': track.get('external_urls', {}),
                'uri': track['uri']
            }
        except Exception as e:
            logger.error(f"Failed to get track details for {track_id}: {e}")
            return {}
    
    async def search_tracks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for tracks using Spotify API."""
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            tracks = []
            
            for track in results['tracks']['items']:
                tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown',
                    'album': track['album']['name'] if track['album'] else 'Unknown',
                    'album_art': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'popularity': track.get('popularity', 0),
                    'uri': track['uri']
                })
            
            return tracks
        except Exception as e:
            logger.error(f"Failed to search tracks with query '{query}': {e}")
            return []
    
    def is_user_authenticated(self, user_id: str) -> bool:
        """Check if user is authenticated."""
        return user_id in self.auth_cache
    
    def get_authenticated_users(self) -> List[str]:
        """Get list of authenticated user IDs."""
        return list(self.auth_cache.keys())
    
    def clear_user_auth(self, user_id: str):
        """Clear authentication for a user."""
        if user_id in self.auth_cache:
            del self.auth_cache[user_id]
