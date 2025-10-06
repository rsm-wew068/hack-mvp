#!/usr/bin/env python3
"""
On-demand YouTube analysis endpoint
Analyzes a YouTube video in real-time and adds to index
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import logging
from youtube_audio_analyzer import YouTubeAudioAnalyzer
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    """Request to analyze a YouTube video"""
    video_id: str
    title: str
    artist: str
    genre: Optional[str] = "Unknown"


class AnalyzeResponse(BaseModel):
    """Analysis result"""
    success: bool
    message: str
    track: Optional[Dict] = None


async def analyze_youtube_video(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a YouTube video on-demand and add to Elasticsearch.
    
    Args:
        request: Analysis request with video details
        
    Returns:
        Analysis response with track data
    """
    try:
        # Initialize analyzer
        analyzer = YouTubeAudioAnalyzer()
        
        # Analyze track
        logger.info(
            f"🎵 Analyzing on-demand: {request.artist} - {request.title}"
        )
        
        result = analyzer.analyze_track(
            request.video_id,
            request.title,
            request.artist
        )
        
        if not result:
            return AnalyzeResponse(
                success=False,
                message="Failed to analyze audio"
            )
        
        # Add genre
        result['genre'] = request.genre
        
        # Connect to Elasticsearch
        es_cloud_id = os.getenv("ELASTIC_CLOUD_ID")
        es_api_key = os.getenv("ELASTIC_API_KEY")
        
        if es_cloud_id and es_api_key:
            es = Elasticsearch(
                cloud_id=es_cloud_id,
                api_key=es_api_key
            )
            
            # Index document
            index_name = "music_tracks"
            doc = {
                "video_id": result["video_id"],
                "title": result["title"],
                "artist": result["artist"],
                "bpm": result["bpm"],
                "key": result["key"],
                "energy": result.get("energy", 0.5),
                "genre": result["genre"],
                "embedding": result["embedding"],
                "youtube_data": result.get("youtube_data", {}),
                "analysis_source": "real_audio_ondemand"
            }
            
            es.index(index=index_name, document=doc)
            es.indices.refresh(index=index_name)
            
            logger.info(f"✓ Indexed track: {result['artist']} - {result['title']}")
        
        return AnalyzeResponse(
            success=True,
            message=f"Successfully analyzed and indexed track",
            track=result
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return AnalyzeResponse(
            success=False,
            message=f"Analysis error: {str(e)}"
        )


# This can be added to main.py as a new endpoint:
"""
@app.post("/api/analyze-youtube", response_model=AnalyzeResponse)
async def analyze_youtube(request: AnalyzeRequest):
    '''
    Analyze a YouTube video on-demand and add to search index.
    This enables real-time expansion of the music catalog.
    '''
    return await analyze_youtube_video(request)
"""

if __name__ == "__main__":
    # Test standalone
    import asyncio
    
    test_request = AnalyzeRequest(
        video_id="dQw4w9WgXcQ",
        title="Never Gonna Give You Up",
        artist="Rick Astley",
        genre="Pop/Dance"
    )
    
    result = asyncio.run(analyze_youtube_video(test_request))
    print(f"Result: {result.message}")
    if result.track:
        print(f"BPM: {result.track['bpm']}")
        print(f"Key: {result.track['key']}")
