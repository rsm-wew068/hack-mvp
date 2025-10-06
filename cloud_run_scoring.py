# Cloud Run Scoring Implementation
# Production-ready recommendation scoring

import numpy as np
from typing import Dict, List, Optional
from ai_explainer import AIExplainer

# Initialize AI explainer
try:
    ai_explainer = AIExplainer()
except Exception as e:
    print(f"Warning: AI explainer initialization failed: {e}")
    ai_explainer = None

def calculate_recommendation_score(
    candidate: Dict,
    seed_track: Dict,
    user_profile: Optional[Dict] = None,
    vertex_ranker: Optional[object] = None
) -> float:
    """
    Calculate recommendation score using hybrid approach.
    
    Args:
        candidate: Candidate track with features
        seed_track: Seed track for similarity
        user_profile: User's preference profile
        vertex_ranker: Optional trained ranker model
    
    Returns:
        Final recommendation score (0.0 to 1.0)
    """
    
    # Base scoring components
    audio_sim = cosine_similarity(
        candidate['openl3'], 
        seed_track['openl3']
    )
    
    bpm_affinity = bpm_affinity_score(
        seed_track['bpm'], 
        candidate['bpm']
    )
    
    key_affinity = key_affinity_score(
        seed_track['key'], 
        candidate['key']
    )
    
    # Novelty and diversity
    artist_penalty = artist_repetition_penalty(
        seed_track['artist'], 
        candidate['artist']
    )
    
    novelty_bonus = calculate_novelty_bonus(candidate, user_profile)
    
    # Base score calculation
    base_score = (
        0.6 * audio_sim +
        0.25 * bpm_affinity +
        0.15 * key_affinity -
        0.10 * artist_penalty +
        novelty_bonus
    )
    
    # Optional: Vertex AI ranker enhancement
    if vertex_ranker and user_profile:
        ranker_score = vertex_ranker.predict([
            audio_sim, bpm_affinity, key_affinity, 
            artist_penalty, novelty_bonus
        ])
        # Blend base score with ranker
        final_score = 0.7 * base_score + 0.3 * ranker_score
    else:
        final_score = base_score
    
    return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def bpm_affinity_score(seed_bpm: float, candidate_bpm: float) -> float:
    """Calculate BPM affinity score."""
    if not seed_bpm or not candidate_bpm:
        return 0.5
    
    bpm_diff = abs(seed_bpm - candidate_bpm)
    # Normalize difference (max 30 BPM difference = 0 score)
    return max(0.0, 1.0 - (bpm_diff / 30.0))


def key_affinity_score(seed_key: str, candidate_key: str) -> float:
    """Calculate key compatibility score."""
    if not seed_key or not candidate_key:
        return 0.5
    
    # Perfect match
    if seed_key == candidate_key:
        return 1.0
    
    # Relative key match (e.g., C major <-> A minor)
    if relative_key_match(seed_key, candidate_key):
        return 0.8
    
    # Same mode (major/minor)
    if same_mode(seed_key, candidate_key):
        return 0.6
    
    return 0.3


def relative_key_match(key1: str, key2: str) -> bool:
    """Check if keys are relative (e.g., C major <-> A minor)."""
    # Simplified relative key matching
    relative_pairs = [
        ("C major", "A minor"), ("A minor", "C major"),
        ("G major", "E minor"), ("E minor", "G major"),
        ("D major", "B minor"), ("B minor", "D major"),
        # Add more relative key pairs as needed
    ]
    
    return (key1, key2) in relative_pairs or (key2, key1) in relative_pairs


def same_mode(key1: str, key2: str) -> bool:
    """Check if both keys are in the same mode (major/minor)."""
    major1 = "major" in key1.lower()
    major2 = "major" in key2.lower()
    return major1 == major2


def artist_repetition_penalty(seed_artist: str, candidate_artist: str) -> float:
    """Calculate penalty for artist repetition."""
    if seed_artist == candidate_artist:
        return 1.0  # Full penalty for same artist
    return 0.0


def calculate_novelty_bonus(candidate: Dict, user_profile: Optional[Dict]) -> float:
    """Calculate novelty bonus for diverse recommendations."""
    if not user_profile:
        return 0.0
    
    # Check if artist is new to user
    user_artists = user_profile.get('listened_artists', set())
    if candidate['artist'] not in user_artists:
        return 0.2  # Bonus for new artist
    
    # Check if genre is new to user
    user_genres = user_profile.get('listened_genres', set())
    candidate_genre = candidate.get('genre', '')
    if candidate_genre and candidate_genre not in user_genres:
        return 0.1  # Bonus for new genre
    
    return 0.0


def generate_explanation(
    candidate: Dict,
    seed_track: Dict,
    score: float,
    use_ai: bool = True
) -> str:
    """
    Generate explanation for recommendation.

    Args:
        candidate: Recommended track
        seed_track: Original seed track
        score: Recommendation score
        use_ai: Whether to use AI-generated explanations

    Returns:
        Natural language explanation
    """
    # Try AI explanation first if enabled
    if use_ai and ai_explainer:
        try:
            # Prepare audio features comparison
            audio_features = {
                'seed_bpm': seed_track.get('bpm', 120),
                'recommended_bpm': candidate.get('bpm', 120),
                'seed_key': seed_track.get('key', 'Unknown'),
                'recommended_key': candidate.get('key', 'Unknown'),
                'seed_energy': seed_track.get('energy', 0.5),
                'recommended_energy': candidate.get('energy', 0.5),
                'seed_danceability': seed_track.get('danceability', 0.5),
                'recommended_danceability': candidate.get('danceability', 0.5)
            }

            ai_explanation = ai_explainer.generate_recommendation_explanation(
                seed_track=seed_track,
                recommended_track=candidate,
                similarity_score=score,
                audio_features=audio_features
            )

            if ai_explanation:
                return ai_explanation
        except Exception as e:
            print(f"AI explanation failed: {e}, falling back to template")

    # Fallback: Generate rationale chips
    chips = []

    # BPM analysis
    if candidate.get('bpm'):
        chips.append(f"{candidate['bpm']:.0f} BPM")

    # Key analysis
    if candidate.get('key'):
        chips.append(candidate['key'])

    # Audio similarity
    if score > 0.8:
        chips.append("high audio similarity")
    elif score > 0.6:
        chips.append("moderate audio similarity")

    # Novelty indicators
    if seed_track.get('artist') != candidate.get('artist'):
        chips.append("new artist")

    return " • ".join(chips) if chips else f"{int(score * 100)}% match"


# Example usage for Cloud Run FastAPI endpoint
def recommend_tracks(
    seed_track: Dict,
    candidates: List[Dict],
    user_profile: Optional[Dict] = None,
    vertex_ranker: Optional[object] = None,
    top_k: int = 10
) -> List[Dict]:
    """
    Generate top-k recommendations with scores and explanations.
    
    Args:
        seed_track: Reference track for similarity
        candidates: List of candidate tracks
        user_profile: User's preference profile
        vertex_ranker: Optional trained ranker
        top_k: Number of recommendations to return
    
    Returns:
        List of scored recommendations with explanations
    """
    
    scored_recommendations = []
    
    for candidate in candidates:
        score = calculate_recommendation_score(
            candidate, seed_track, user_profile, vertex_ranker
        )

        explanation = generate_explanation(
            candidate=candidate,
            seed_track=seed_track,
            score=score
        )
        
        scored_recommendations.append({
            'track': candidate,
            'score': score,
            'explanation': explanation
        })
    
    # Sort by score and return top-k
    return sorted(
        scored_recommendations, 
        key=lambda x: x['score'], 
        reverse=True
    )[:top_k]
