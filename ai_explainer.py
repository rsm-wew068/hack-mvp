#!/usr/bin/env python3
"""
AI Explanation Generator using Vertex AI Gemini
Generates natural language explanations for music recommendations
"""

import os
from typing import Dict, List, Optional
import json
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from dotenv import load_dotenv

load_dotenv()

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'goole-hackathon')
LOCATION = 'us-central1'
MODEL_NAME = 'gemini-1.5-flash'  # Fast and efficient for explanations

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)


class AIExplainer:
    """Generate AI-powered explanations for music recommendations"""
    
    def __init__(self):
        """Initialize the Gemini model"""
        self.model = GenerativeModel(MODEL_NAME)
        self.generation_config = GenerationConfig(
            temperature=0.7,  # Creative but consistent
            top_p=0.9,
            top_k=40,
            max_output_tokens=150,  # Short, concise explanations
        )
    
    def generate_recommendation_explanation(
        self,
        seed_track: Dict,
        recommended_track: Dict,
        similarity_score: float,
        audio_features: Dict
    ) -> str:
        """
        Generate natural language explanation for why a track is recommended
        
        Args:
            seed_track: The seed track user selected
            recommended_track: The recommended track
            similarity_score: Cosine similarity score (0-1)
            audio_features: Audio feature comparison (BPM, key, energy)
        
        Returns:
            Natural language explanation string
        """
        
        # Create prompt for Gemini
        prompt = f"""You're a cool music DJ talking to a friend. Explain in ONE casual, friendly sentence why they'll love this song.

They're listening to: "{seed_track['title']}" by {seed_track['artist']}
You're recommending: "{recommended_track['title']}" by {recommended_track['artist']}

Musical vibes:
- Same tempo: {abs(audio_features.get('bpm_diff', 0)) < 10}
- Same key: {audio_features.get('key_match', 'different') == 'same'}
- Same genre: {seed_track.get('genre') == recommended_track.get('genre')}
- Match strength: {similarity_score:.0%}

Write like a friend texting, not a robot. Use emojis. No technical terms!
Examples: "🔥 This has the same chill vibe!" or "Perfect match - same energy and groove!"
Keep it under 100 characters."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            explanation = response.text.strip()
            
            # Ensure it's not too long
            if len(explanation) > 200:
                explanation = explanation[:197] + "..."
            
            return explanation
        
        except Exception as e:
            print(f"⚠️  Error generating AI explanation: {e}")
            # Fallback to template-based explanation
            return self._fallback_explanation(
                seed_track, recommended_track, similarity_score, audio_features
            )
    
    def generate_playlist_explanation(
        self,
        query: str,
        tracks: List[Dict],
        mood: Optional[str] = None
    ) -> str:
        """
        Generate explanation for a text-to-playlist recommendation
        
        Args:
            query: User's search query (e.g., "lo-fi beats for studying")
            tracks: List of recommended tracks
            mood: Optional mood/vibe detected
        
        Returns:
            Playlist explanation string
        """
        
        # Extract track details
        artists = list(set(t['artist'] for t in tracks[:5]))
        genres = list(set(t.get('genre', 'Unknown') for t in tracks if t.get('genre')))
        avg_bpm = sum(t.get('bpm', 120) for t in tracks) / len(tracks)
        
        prompt = f"""You are a music curator AI. Explain in 2-3 SHORT sentences why this playlist matches the user's request.

User Query: "{query}"
Detected Mood: {mood or 'general'}

Playlist Details:
- {len(tracks)} tracks
- Artists: {', '.join(artists[:3])}...
- Genres: {', '.join(genres[:3]) if genres else 'Various'}
- Average BPM: {avg_bpm:.0f}

Write a friendly, engaging explanation of why this playlist fits their request.
Make it personal and exciting. Keep it under 150 characters if possible."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text.strip()
        
        except Exception as e:
            print(f"⚠️  Error generating playlist explanation: {e}")
            return self._fallback_playlist_explanation(query, tracks, genres, avg_bpm)
    
    def _fallback_explanation(
        self,
        seed_track: Dict,
        rec_track: Dict,
        score: float,
        features: Dict
    ) -> str:
        """User-friendly fallback if Gemini fails"""
        
        # Get track details
        bpm_diff = abs(features.get('bpm_diff', 0))
        key_match = features.get('key_match') == 'same'
        same_genre = seed_track.get('genre') == rec_track.get('genre')
        genre = rec_track.get('genre', 'track')
        
        # Create friendly explanations
        if same_genre and bpm_diff < 10 and key_match:
            return f"🎵 Perfect match! Same {genre} vibe with matching tempo and key!"
        elif same_genre and bpm_diff < 10:
            return f"🔥 Great {genre} pick with the same energy and groove!"
        elif same_genre:
            return f"✨ Another awesome {genre} track you'll love!"
        elif bpm_diff < 10 and key_match:
            return "🎧 Same tempo and key - flows perfectly together!"
        elif bpm_diff < 10:
            return "💫 Same energy level, great for keeping the vibe going!"
        elif key_match:
            return "🎹 Harmonically compatible - sounds great together!"
        elif score > 0.95:
            return "🌟 Super similar sound - you'll definitely dig this!"
        else:
            return f"👍 Strong match with similar musical style!"
    
    def _fallback_playlist_explanation(
        self,
        query: str,
        tracks: List[Dict],
        genres: List[str],
        avg_bpm: float
    ) -> str:
        """User-friendly fallback for playlist"""
        
        if genres and len(genres) > 0:
            genre = genres[0]
            return f"🎵 {len(tracks)} {genre} tracks perfect for '{query}'!"
        else:
            return f"🎧 {len(tracks)} hand-picked tracks for '{query}'!"


# Singleton instance
_explainer = None

def get_explainer() -> AIExplainer:
    """Get or create the AI explainer instance"""
    global _explainer
    if _explainer is None:
        _explainer = AIExplainer()
    return _explainer


def generate_explanation(
    seed_track: Dict,
    recommended_track: Dict,
    similarity_score: float,
    audio_features: Dict
) -> str:
    """
    Convenience function to generate explanation
    
    Usage:
        explanation = generate_explanation(
            seed_track={'title': 'The Navy Song', 'artist': 'Billy Talent', ...},
            recommended_track={'title': 'Birthday Sex', 'artist': 'Jeremih', ...},
            similarity_score=0.76,
            audio_features={'bpm_diff': 0.4, 'key_match': 'different', ...}
        )
    """
    explainer = get_explainer()
    return explainer.generate_recommendation_explanation(
        seed_track, recommended_track, similarity_score, audio_features
    )


def generate_playlist_explanation(
    query: str,
    tracks: List[Dict],
    mood: Optional[str] = None
) -> str:
    """
    Convenience function to generate playlist explanation
    
    Usage:
        explanation = generate_playlist_explanation(
            query="lo-fi beats for studying",
            tracks=[{...}, {...}, ...],
            mood="relaxed"
        )
    """
    explainer = get_explainer()
    return explainer.generate_playlist_explanation(query, tracks, mood)


# Test function
def test_explainer():
    """Test the AI explainer with sample data"""
    print("🧪 Testing AI Explainer...\n")
    
    seed = {
        'title': 'The Navy Song',
        'artist': 'Billy Talent',
        'bpm': 89,
        'key': 'D# minor',
        'genre': 'Rock'
    }
    
    rec = {
        'title': 'Birthday Sex',
        'artist': 'Jeremih',
        'bpm': 84,
        'key': 'C# minor',
        'genre': 'R&B'
    }
    
    features = {
        'bpm_diff': 5,
        'key_match': 'related',
        'energy_match': 'similar'
    }
    
    print("Generating recommendation explanation...")
    explanation = generate_explanation(seed, rec, 0.76, features)
    print(f"✨ {explanation}\n")
    
    print("Generating playlist explanation...")
    playlist_exp = generate_playlist_explanation(
        "lo-fi beats for studying",
        [seed, rec],
        "relaxed"
    )
    print(f"✨ {playlist_exp}\n")
    
    print("✅ AI Explainer test complete!")


if __name__ == '__main__':
    test_explainer()
