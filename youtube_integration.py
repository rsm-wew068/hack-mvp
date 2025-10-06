#!/usr/bin/env python3
"""
YouTube API Integration
Fetches video metadata for music tracks
"""

import os
from typing import Dict, Optional, List
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from dotenv import load_dotenv
import time

load_dotenv()

# YouTube API Configuration
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
CREDENTIALS_PATH = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')


class YouTubeClient:
    """Client for YouTube Data API v3"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize YouTube API client using Google Cloud credentials
        
        Args:
            credentials_path: Path to service account credentials
                            (optional, reads from GOOGLE_APPLICATION_CREDENTIALS)
        """
        self.credentials_path = credentials_path or CREDENTIALS_PATH
        self.youtube = None
        
        if self.credentials_path and os.path.exists(self.credentials_path):
            try:
                # Load service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/youtube.readonly']
                )
                
                self.youtube = build(
                    YOUTUBE_API_SERVICE_NAME,
                    YOUTUBE_API_VERSION,
                    credentials=credentials
                )
                print("✅ YouTube API client initialized with service account")
            except Exception as e:
                print(f"⚠️  YouTube API initialization failed: {e}")
                print("    Continuing without YouTube integration")
                self.youtube = None
        else:
            print("⚠️  Google Cloud credentials not found")
            print(f"    Looking for: {self.credentials_path}")
            print("    YouTube integration will be disabled")
    
    def search_music_video(
        self,
        title: str,
        artist: str,
        max_results: int = 1
    ) -> Optional[Dict]:
        """
        Search for a music video on YouTube
        
        Args:
            title: Track title
            artist: Artist name
            max_results: Number of results to return
        
        Returns:
            Video metadata dict or None
        """
        if not self.youtube:
            return None
        
        try:
            # Build search query
            query = f"{artist} {title} official music video"
            
            # Search for videos
            search_response = self.youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=max_results,
                type='video',
                videoCategoryId='10',  # Music category
                order='relevance'
            ).execute()
            
            if not search_response.get('items'):
                return None
            
            # Get first result
            video = search_response['items'][0]
            video_id = video['id']['videoId']
            
            # Get video details (statistics, duration)
            video_details = self.youtube.videos().list(
                part='statistics,contentDetails,snippet',
                id=video_id
            ).execute()
            
            if not video_details.get('items'):
                return None
            
            details = video_details['items'][0]
            
            return {
                'video_id': video_id,
                'title': details['snippet']['title'],
                'channel': details['snippet']['channelTitle'],
                'thumbnail_default': details['snippet']['thumbnails']['default']['url'],
                'thumbnail_medium': details['snippet']['thumbnails']['medium']['url'],
                'thumbnail_high': details['snippet']['thumbnails']['high']['url'],
                'view_count': int(details['statistics'].get('viewCount', 0)),
                'like_count': int(details['statistics'].get('likeCount', 0)),
                'duration': details['contentDetails']['duration'],
                'published_at': details['snippet']['publishedAt'],
                'embed_url': f"https://www.youtube.com/embed/{video_id}",
                'watch_url': f"https://www.youtube.com/watch?v={video_id}"
            }
            
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return None
        except Exception as e:
            print(f"Error searching YouTube: {e}")
            return None
    
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """
        Get details for a specific video ID
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            Video metadata dict or None
        """
        if not self.youtube:
            return None
        
        try:
            video_details = self.youtube.videos().list(
                part='statistics,contentDetails,snippet',
                id=video_id
            ).execute()
            
            if not video_details.get('items'):
                return None
            
            details = video_details['items'][0]
            
            return {
                'video_id': video_id,
                'title': details['snippet']['title'],
                'channel': details['snippet']['channelTitle'],
                'thumbnail_default': details['snippet']['thumbnails']['default']['url'],
                'thumbnail_medium': details['snippet']['thumbnails']['medium']['url'],
                'thumbnail_high': details['snippet']['thumbnails']['high']['url'],
                'view_count': int(details['statistics'].get('viewCount', 0)),
                'like_count': int(details['statistics'].get('likeCount', 0)),
                'duration': details['contentDetails']['duration'],
                'published_at': details['snippet']['publishedAt'],
                'embed_url': f"https://www.youtube.com/embed/{video_id}",
                'watch_url': f"https://www.youtube.com/watch?v={video_id}"
            }
            
        except HttpError as e:
            print(f"YouTube API error: {e}")
            return None
        except Exception as e:
            print(f"Error getting video details: {e}")
            return None
    
    def bulk_search_tracks(
        self,
        tracks: List[Dict],
        delay: float = 0.1
    ) -> Dict[str, Dict]:
        """
        Search YouTube for multiple tracks
        
        Args:
            tracks: List of track dicts with 'title' and 'artist'
            delay: Delay between requests (rate limiting)
        
        Returns:
            Dict mapping track_id to video metadata
        """
        results = {}
        
        for track in tracks:
            track_id = track.get('track_id')
            title = track.get('title')
            artist = track.get('artist')
            
            if not all([track_id, title, artist]):
                continue
            
            print(f"Searching YouTube: {artist} - {title}")
            
            video_data = self.search_music_video(title, artist)
            
            if video_data:
                results[track_id] = video_data
                print(f"  ✅ Found: {video_data['video_id']}")
            else:
                print(f"  ❌ Not found")
            
            # Rate limiting
            time.sleep(delay)
        
        return results


def enrich_track_with_youtube(track: Dict) -> Dict:
    """
    Enrich a track dict with YouTube metadata
    
    Args:
        track: Track dict with title, artist
    
    Returns:
        Track dict with youtube_data field
    """
    youtube_client = YouTubeClient()
    
    # If track already has youtube_video_id, get details
    if track.get('yt_video_id'):
        video_data = youtube_client.get_video_details(track['yt_video_id'])
        if video_data:
            track['youtube_data'] = video_data
            return track
    
    # Otherwise search for video
    video_data = youtube_client.search_music_video(
        title=track.get('title', ''),
        artist=track.get('artist', '')
    )
    
    if video_data:
        track['youtube_data'] = video_data
        track['yt_video_id'] = video_data['video_id']
    
    return track


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing YouTube API Integration")
    print("=" * 60)
    
    youtube = YouTubeClient()
    
    if youtube.youtube:
        print("\n📺 Test 1: Search for music video")
        result = youtube.search_music_video(
            title="Bohemian Rhapsody",
            artist="Queen"
        )
        
        if result:
            print(f"✅ Found video:")
            print(f"   Video ID: {result['video_id']}")
            print(f"   Title: {result['title']}")
            print(f"   Channel: {result['channel']}")
            print(f"   Views: {result['view_count']:,}")
            print(f"   Watch: {result['watch_url']}")
        else:
            print("❌ No video found")
        
        print("\n📺 Test 2: Enrich track with YouTube data")
        test_track = {
            'track_id': '000002',
            'title': 'Bohemian Rhapsody',
            'artist': 'Queen'
        }
        
        enriched = enrich_track_with_youtube(test_track)
        
        if enriched.get('youtube_data'):
            print(f"✅ Track enriched:")
            print(f"   YouTube ID: {enriched['yt_video_id']}")
            yt_data = enriched['youtube_data']
            print(f"   Thumbnail: {yt_data['thumbnail_medium']}")
        else:
            print("❌ Could not enrich track")
    else:
        print("\n⚠️  YouTube API not available")
        print("Ensure Google Cloud credentials are configured:")
        print("  1. credentials.json exists in project root")
        print("  2. GOOGLE_APPLICATION_CREDENTIALS is set")
        print("  3. YouTube Data API v3 is enabled in GCP project")
    
    print("\n" + "=" * 60)
