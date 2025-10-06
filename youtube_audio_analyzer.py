#!/usr/bin/env python3
"""
YouTube Audio Analysis Pipeline
Analyzes audio from YouTube videos to extract music features.
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Optional
import numpy as np
import subprocess

# Audio analysis libraries
try:
    import librosa
    import openl3
    import soundfile
except ImportError:
    print("Installing audio analysis dependencies...")
    subprocess.run(["pip", "install", "librosa", "openl3", "soundfile"], check=True)
    import librosa
    import openl3
    import soundfile

from youtube_integration import YouTubeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeAudioAnalyzer:
    """Analyze audio from YouTube videos."""
    
    def __init__(self):
        self.youtube_client = YouTubeClient()
        
    def download_audio(self, video_id: str, output_path: str) -> bool:
        """
        Download audio from YouTube video using yt-dlp.
        
        Args:
            video_id: YouTube video ID
            output_path: Path to save audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use yt-dlp to download audio only
            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", "wav",  # Convert to WAV
                "--audio-quality", "0",  # Best quality
                "-o", output_path,
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Downloaded audio for video {video_id}")
                return True
            else:
                logger.error(f"Failed to download {video_id}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Download timeout for video {video_id}")
            return False
        except Exception as e:
            logger.error(f"Download error for {video_id}: {e}")
            return False
    
    def analyze_bpm(self, audio_path: str) -> Optional[float]:
        """
        Extract BPM (tempo) from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            BPM value or None if failed
        """
        try:
            y, sr = librosa.load(audio_path, duration=30.0)  # Analyze first 30 seconds
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Handle numpy array if returned
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else float(tempo)
            
            logger.info(f"Detected BPM: {tempo:.1f}")
            return float(tempo)
            
        except Exception as e:
            logger.error(f"BPM analysis error: {e}")
            return None
    
    def analyze_key(self, audio_path: str) -> Optional[str]:
        """
        Detect musical key from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Key string (e.g., "C major") or None if failed
        """
        try:
            y, sr = librosa.load(audio_path, duration=30.0)
            
            # Compute chroma features
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Average over time
            chroma_mean = np.mean(chroma, axis=1)
            
            # Find dominant pitch class
            key_index = np.argmax(chroma_mean)
            
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Simple major/minor detection based on chroma distribution
            # (This is a heuristic - real key detection is complex)
            minor_third = chroma_mean[(key_index + 3) % 12]
            major_third = chroma_mean[(key_index + 4) % 12]
            
            mode = "major" if major_third > minor_third else "minor"
            key = f"{keys[key_index]} {mode}"
            
            logger.info(f"Detected key: {key}")
            return key
            
        except Exception as e:
            logger.error(f"Key detection error: {e}")
            return None
    
    def analyze_energy(self, audio_path: str) -> Optional[float]:
        """
        Calculate energy/loudness of audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Energy value 0-1 or None if failed
        """
        try:
            y, sr = librosa.load(audio_path, duration=30.0)
            
            # RMS energy
            rms = librosa.feature.rms(y=y)
            energy = float(np.mean(rms))
            
            # Normalize to 0-1 range (heuristic)
            energy_normalized = min(energy * 10, 1.0)
            
            logger.info(f"Detected energy: {energy_normalized:.3f}")
            return energy_normalized
            
        except Exception as e:
            logger.error(f"Energy analysis error: {e}")
            return None
    
    def generate_embedding(self, audio_path: str) -> Optional[List[float]]:
        """
        Generate OpenL3 music embedding.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            512-dimensional embedding vector or None if failed
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=48000, duration=30.0)
            
            # Generate OpenL3 embedding (music model, 512-dim)
            embedding, _ = openl3.get_audio_embedding(
                y, sr,
                content_type="music",
                embedding_size=512
            )
            
            # Average over time windows to get single 512-dim vector
            embedding_avg = np.mean(embedding, axis=0)
            
            logger.info(f"Generated OpenL3 embedding: {embedding_avg.shape}")
            return embedding_avg.tolist()
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None
    
    def analyze_track(self, video_id: str, title: str, artist: str) -> Optional[Dict]:
        """
        Complete analysis pipeline for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            title: Track title
            artist: Artist name
            
        Returns:
            Dictionary with audio features or None if failed
        """
        logger.info(f"Analyzing: {artist} - {title} ({video_id})")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = os.path.join(tmp_dir, f"{video_id}.wav")
            
            # Download audio
            if not self.download_audio(video_id, audio_path):
                return None
            
            # Check if file exists and has size
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
                logger.error(f"Downloaded file is invalid: {audio_path}")
                return None
            
            # Analyze audio features
            bpm = self.analyze_bpm(audio_path)
            key = self.analyze_key(audio_path)
            energy = self.analyze_energy(audio_path)
            embedding = self.generate_embedding(audio_path)
            
            if not all([bpm, key, embedding]):
                logger.warning(f"Incomplete analysis for {video_id}")
                return None
            
            # Get YouTube metadata
            yt_data = self.youtube_client.get_video_details(video_id)
            
            result = {
                "video_id": video_id,
                "title": title,
                "artist": artist,
                "bpm": round(bpm, 1),
                "key": key,
                "energy": round(energy, 3) if energy else 0.5,
                "embedding": embedding,
                "youtube_data": yt_data,
                "analysis_source": "real_audio"
            }
            
            logger.info(f"✓ Successfully analyzed {artist} - {title}")
            return result


def analyze_curated_list(video_list_path: str, output_path: str):
    """
    Analyze a list of curated YouTube videos.
    
    Args:
        video_list_path: Path to JSON file with video list
        output_path: Path to save analyzed tracks
    """
    analyzer = YouTubeAudioAnalyzer()
    
    # Load video list
    with open(video_list_path, 'r') as f:
        videos = json.load(f)
    
    logger.info(f"Analyzing {len(videos)} videos...")
    
    analyzed_tracks = []
    failed = []
    
    for i, video in enumerate(videos, 1):
        logger.info(f"[{i}/{len(videos)}] Processing {video['artist']} - {video['title']}")
        
        result = analyzer.analyze_track(
            video['video_id'],
            video['title'],
            video['artist']
        )
        
        if result:
            analyzed_tracks.append(result)
        else:
            failed.append(video)
        
        # Save progress every 10 tracks
        if i % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump(analyzed_tracks, f, indent=2)
            logger.info(f"Progress saved: {len(analyzed_tracks)} analyzed, {len(failed)} failed")
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump(analyzed_tracks, f, indent=2)
    
    logger.info(f"✓ Analysis complete!")
    logger.info(f"  Successful: {len(analyzed_tracks)}")
    logger.info(f"  Failed: {len(failed)}")
    
    if failed:
        failed_path = output_path.replace('.json', '_failed.json')
        with open(failed_path, 'w') as f:
            json.dump(failed, f, indent=2)
        logger.info(f"  Failed videos saved to: {failed_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python youtube_audio_analyzer.py <video_list.json> <output.json>")
        print("Example: python youtube_audio_analyzer.py curated_videos.json analyzed_tracks.json")
        sys.exit(1)
    
    analyze_curated_list(sys.argv[1], sys.argv[2])
