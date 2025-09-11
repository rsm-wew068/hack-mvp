"""Tests for Spotify integration."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from integrations.spotify_client import SpotifyClient


class TestSpotifyClient:
    """Test cases for Spotify client."""
    
    @pytest.fixture
    def spotify_client(self):
        """Create Spotify client with mocked credentials."""
        with patch('integrations.spotify_client.settings') as mock_settings:
            mock_settings.spotify_client_id = "test_client_id"
            mock_settings.spotify_client_secret = "test_client_secret"
            mock_settings.spotify_redirect_uri = "http://localhost:8501/callback"
            
            client = SpotifyClient()
            return client
    
    def test_oauth_url_generation(self, spotify_client):
        """Test OAuth URL generation."""
        user_id = "test_user"
        auth_url = spotify_client.get_oauth_url(user_id)
        
        assert "spotify.com" in auth_url
        assert "client_id" in auth_url
        assert "redirect_uri" in auth_url
        assert "scope" in auth_url
    
    @pytest.mark.asyncio
    async def test_handle_oauth_callback(self, spotify_client):
        """Test OAuth callback handling."""
        user_id = "test_user"
        code = "test_auth_code"
        
        # Mock the OAuth flow
        with patch('integrations.spotify_client.SpotifyOAuth') as mock_oauth:
            mock_auth_manager = Mock()
            mock_auth_manager.get_access_token.return_value = {
                'access_token': 'test_token',
                'expires_in': 3600
            }
            mock_oauth.return_value = mock_auth_manager
            
            token_info = await spotify_client.handle_oauth_callback(user_id, code)
            
            assert 'access_token' in token_info
            assert token_info['access_token'] == 'test_token'
            assert user_id in spotify_client.auth_cache
    
    @pytest.mark.asyncio
    async def test_get_user_info(self, spotify_client):
        """Test getting user information."""
        user_id = "test_user"
        
        # Mock authenticated user
        mock_sp = Mock()
        mock_sp.current_user.return_value = {
            'id': 'test_user',
            'display_name': 'Test User',
            'followers': {'total': 100},
            'country': 'US'
        }
        spotify_client.auth_cache[user_id] = mock_sp
        
        user_info = await spotify_client.get_user_info(user_id)
        
        assert user_info['id'] == 'test_user'
        assert user_info['display_name'] == 'Test User'
        assert user_info['followers'] == 100
        assert user_info['country'] == 'US'
    
    @pytest.mark.asyncio
    async def test_get_user_playlists(self, spotify_client):
        """Test getting user playlists."""
        user_id = "test_user"
        
        # Mock authenticated user
        mock_sp = Mock()
        mock_sp.current_user_playlists.return_value = {
            'items': [
                {
                    'id': 'playlist_1',
                    'name': 'Test Playlist 1',
                    'description': 'Test description',
                    'tracks': {'total': 10},
                    'public': True,
                    'owner': {'display_name': 'Test User'}
                },
                {
                    'id': 'playlist_2',
                    'name': 'Test Playlist 2',
                    'description': '',
                    'tracks': {'total': 5},
                    'public': False,
                    'owner': {'display_name': 'Test User'}
                }
            ],
            'next': None
        }
        spotify_client.auth_cache[user_id] = mock_sp
        
        playlists = await spotify_client.get_user_playlists(user_id)
        
        assert len(playlists) == 2
        assert playlists[0]['id'] == 'playlist_1'
        assert playlists[0]['name'] == 'Test Playlist 1'
        assert playlists[0]['tracks_count'] == 10
        assert playlists[1]['public'] == False
    
    @pytest.mark.asyncio
    async def test_get_playlist_tracks(self, spotify_client):
        """Test getting playlist tracks."""
        playlist_id = "test_playlist"
        user_id = "test_user"
        
        # Mock Spotify client
        mock_sp = Mock()
        mock_sp.playlist_tracks.return_value = {
            'items': [
                {
                    'track': {
                        'id': 'track_1',
                        'name': 'Test Track 1',
                        'artists': [{'name': 'Test Artist 1'}],
                        'album': {'name': 'Test Album 1'},
                        'uri': 'spotify:track:track_1',
                        'duration_ms': 180000,
                        'popularity': 80,
                        'explicit': False
                    }
                },
                {
                    'track': {
                        'id': 'track_2',
                        'name': 'Test Track 2',
                        'artists': [{'name': 'Test Artist 2'}],
                        'album': {'name': 'Test Album 2'},
                        'uri': 'spotify:track:track_2',
                        'duration_ms': 200000,
                        'popularity': 70,
                        'explicit': True
                    }
                }
            ],
            'next': None
        }
        spotify_client.auth_cache[user_id] = mock_sp
        
        tracks = await spotify_client.get_playlist_tracks(playlist_id, user_id)
        
        assert len(tracks) == 2
        assert tracks[0]['id'] == 'track_1'
        assert tracks[0]['name'] == 'Test Track 1'
        assert tracks[0]['artist'] == 'Test Artist 1'
        assert tracks[0]['explicit'] == False
        assert tracks[1]['explicit'] == True
    
    @pytest.mark.asyncio
    async def test_get_audio_features(self, spotify_client):
        """Test getting audio features."""
        track_id = "test_track"
        
        # Mock audio features response
        mock_features = {
            'danceability': 0.8,
            'energy': 0.7,
            'valence': 0.6,
            'tempo': 120.0,
            'acousticness': 0.1,
            'instrumentalness': 0.0,
            'speechiness': 0.05,
            'liveness': 0.1,
            'loudness': -6.0,
            'key': 0,
            'mode': 1,
            'time_signature': 4
        }
        
        spotify_client.sp.audio_features.return_value = [mock_features]
        
        features = await spotify_client.get_audio_features(track_id)
        
        assert features['danceability'] == 0.8
        assert features['energy'] == 0.7
        assert features['valence'] == 0.6
        assert features['tempo'] == 120.0
        assert features['acousticness'] == 0.1
        assert features['instrumentalness'] == 0.0
        assert features['speechiness'] == 0.05
        assert features['liveness'] == 0.1
        assert features['loudness'] == -6.0
        assert features['key'] == 0
        assert features['mode'] == 1
        assert features['time_signature'] == 4
    
    @pytest.mark.asyncio
    async def test_get_track_details(self, spotify_client):
        """Test getting track details."""
        track_id = "test_track"
        
        # Mock track details response
        mock_track = {
            'id': 'test_track',
            'name': 'Test Track',
            'artists': [{'name': 'Test Artist'}],
            'album': {
                'name': 'Test Album',
                'images': [{'url': 'https://example.com/art.jpg'}]
            },
            'duration_ms': 180000,
            'popularity': 80,
            'explicit': False,
            'preview_url': 'https://example.com/preview.mp3',
            'external_urls': {'spotify': 'https://open.spotify.com/track/test_track'},
            'uri': 'spotify:track:test_track'
        }
        
        spotify_client.sp.track.return_value = mock_track
        
        track_details = await spotify_client.get_track_details(track_id)
        
        assert track_details['id'] == 'test_track'
        assert track_details['name'] == 'Test Track'
        assert track_details['artist'] == 'Test Artist'
        assert track_details['album'] == 'Test Album'
        assert track_details['album_art'] == 'https://example.com/art.jpg'
        assert track_details['duration_ms'] == 180000
        assert track_details['popularity'] == 80
        assert track_details['explicit'] == False
        assert track_details['preview_url'] == 'https://example.com/preview.mp3'
        assert track_details['spotify_url'] == 'https://open.spotify.com/track/test_track'
    
    @pytest.mark.asyncio
    async def test_search_tracks(self, spotify_client):
        """Test searching tracks."""
        query = "test query"
        limit = 5
        
        # Mock search response
        mock_search_results = {
            'tracks': {
                'items': [
                    {
                        'id': 'track_1',
                        'name': 'Test Track 1',
                        'artists': [{'name': 'Test Artist 1'}],
                        'album': {
                            'name': 'Test Album 1',
                            'images': [{'url': 'https://example.com/art1.jpg'}]
                        },
                        'popularity': 80,
                        'uri': 'spotify:track:track_1'
                    },
                    {
                        'id': 'track_2',
                        'name': 'Test Track 2',
                        'artists': [{'name': 'Test Artist 2'}],
                        'album': {
                            'name': 'Test Album 2',
                            'images': [{'url': 'https://example.com/art2.jpg'}]
                        },
                        'popularity': 70,
                        'uri': 'spotify:track:track_2'
                    }
                ]
            }
        }
        
        spotify_client.sp.search.return_value = mock_search_results
        
        tracks = await spotify_client.search_tracks(query, limit)
        
        assert len(tracks) == 2
        assert tracks[0]['id'] == 'track_1'
        assert tracks[0]['name'] == 'Test Track 1'
        assert tracks[0]['artist'] == 'Test Artist 1'
        assert tracks[0]['album_art'] == 'https://example.com/art1.jpg'
        assert tracks[0]['popularity'] == 80
    
    def test_is_user_authenticated(self, spotify_client):
        """Test user authentication check."""
        user_id = "test_user"
        
        # Initially not authenticated
        assert not spotify_client.is_user_authenticated(user_id)
        
        # Add to auth cache
        spotify_client.auth_cache[user_id] = Mock()
        assert spotify_client.is_user_authenticated(user_id)
    
    def test_get_authenticated_users(self, spotify_client):
        """Test getting authenticated users."""
        # Initially empty
        assert spotify_client.get_authenticated_users() == []
        
        # Add users
        spotify_client.auth_cache["user1"] = Mock()
        spotify_client.auth_cache["user2"] = Mock()
        
        users = spotify_client.get_authenticated_users()
        assert len(users) == 2
        assert "user1" in users
        assert "user2" in users
    
    def test_clear_user_auth(self, spotify_client):
        """Test clearing user authentication."""
        user_id = "test_user"
        
        # Add user to cache
        spotify_client.auth_cache[user_id] = Mock()
        assert spotify_client.is_user_authenticated(user_id)
        
        # Clear authentication
        spotify_client.clear_user_auth(user_id)
        assert not spotify_client.is_user_authenticated(user_id)


if __name__ == "__main__":
    pytest.main([__file__])
