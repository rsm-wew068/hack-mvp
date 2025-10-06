"""
Conversational AI for natural language music search.
Converts queries like "lo-fi beats for studying" to structured search parameters.
"""

# Mood → Genre + BPM mappings
MOOD_MAPPINGS = {
    # Study/Focus moods
    "study": {
        "genres": ["Lo-Fi", "Ambient", "Classical", "Electronic"],
        "bpm_range": (60, 90),
        "keywords": ["chill", "calm", "focus", "concentration"]
    },
    "focus": {
        "genres": ["Ambient", "Classical", "Lo-Fi"],
        "bpm_range": (60, 80),
        "keywords": ["minimal", "quiet", "peaceful"]
    },
    
    # Energy moods
    "workout": {
        "genres": ["Electronic", "Hip-Hop", "Rock", "Pop"],
        "bpm_range": (120, 160),
        "keywords": ["energy", "pump", "intense", "power"]
    },
    "party": {
        "genres": ["Electronic", "Hip-Hop", "Pop", "Dance"],
        "bpm_range": (110, 140),
        "keywords": ["dance", "club", "beat", "fun"]
    },
    
    # Relax moods
    "relax": {
        "genres": ["Ambient", "Jazz", "Classical", "Acoustic"],
        "bpm_range": (50, 80),
        "keywords": ["chill", "calm", "peaceful", "soft"]
    },
    "sleep": {
        "genres": ["Ambient", "Classical"],
        "bpm_range": (40, 70),
        "keywords": ["quiet", "gentle", "soothing", "calm"]
    },
    
    # Social moods
    "dinner": {
        "genres": ["Jazz", "Soul", "R&B", "Acoustic"],
        "bpm_range": (70, 100),
        "keywords": ["smooth", "elegant", "sophisticated"]
    },
    "drive": {
        "genres": ["Rock", "Electronic", "Hip-Hop"],
        "bpm_range": (90, 130),
        "keywords": ["groovy", "rhythmic", "steady"]
    }
}

# Genre synonyms and variations
GENRE_SYNONYMS = {
    "hip hop": ["Hip-Hop", "Rap", "Urban"],
    "electronic": ["Electronic", "EDM", "Techno", "House"],
    "rock": ["Rock", "Alternative", "Indie"],
    "chill": ["Lo-Fi", "Ambient", "Chillout"],
    "classical": ["Classical", "Symphony", "Orchestra"],
    "jazz": ["Jazz", "Blues", "Soul"],
    "pop": ["Pop", "Top 40"],
}

# Activity → Mood mappings
ACTIVITY_MAPPINGS = {
    "studying": "study",
    "working": "focus",
    "exercising": "workout",
    "running": "workout",
    "gym": "workout",
    "party": "party",
    "dancing": "party",
    "sleeping": "sleep",
    "relaxing": "relax",
    "chilling": "relax",
    "cooking": "dinner",
    "driving": "drive",
}


def understand_query(query: str) -> dict:
    """
    Parse natural language query into structured search parameters.
    
    Args:
        query: Natural language query like "lo-fi beats for studying"
        
    Returns:
        dict with genres, bpm_range, keywords, and elasticsearch_query
    """
    query_lower = query.lower()
    
    result = {
        "original_query": query,
        "genres": [],
        "bpm_range": None,
        "keywords": [],
        "mood": None,
        "activity": None,
        "elasticsearch_query": query  # Default to original
    }
    
    # 1. Check for activities
    for activity, mood in ACTIVITY_MAPPINGS.items():
        if activity in query_lower:
            result["activity"] = activity
            result["mood"] = mood
            mood_data = MOOD_MAPPINGS.get(mood, {})
            result["genres"].extend(mood_data.get("genres", []))
            result["bpm_range"] = mood_data.get("bpm_range")
            result["keywords"].extend(mood_data.get("keywords", []))
            break
    
    # 2. Check for explicit moods
    if not result["mood"]:
        for mood, mood_data in MOOD_MAPPINGS.items():
            if mood in query_lower:
                result["mood"] = mood
                result["genres"].extend(mood_data.get("genres", []))
                result["bpm_range"] = mood_data.get("bpm_range")
                result["keywords"].extend(mood_data.get("keywords", []))
                break
    
    # 3. Check for genre mentions
    for genre_keyword, genres in GENRE_SYNONYMS.items():
        if genre_keyword in query_lower:
            result["genres"].extend(genres)
    
    # 4. Extract genre keywords directly
    common_genres = [
        "rock", "pop", "jazz", "classical", "electronic", 
        "hip-hop", "rap", "ambient", "lo-fi", "indie"
    ]
    for genre in common_genres:
        if genre in query_lower:
            result["genres"].append(genre.title())
    
    # 5. Build enhanced Elasticsearch query
    query_parts = [query]  # Start with original
    
    if result["genres"]:
        # Add genre terms
        query_parts.extend(result["genres"])
    
    if result["keywords"]:
        # Add mood keywords
        query_parts.extend(result["keywords"][:3])  # Top 3 keywords
    
    result["elasticsearch_query"] = " ".join(set(query_parts))
    
    # Remove duplicates
    result["genres"] = list(set(result["genres"]))
    result["keywords"] = list(set(result["keywords"]))
    
    return result


def enhance_search_params(understood_query: dict) -> dict:
    """
    Convert understood query to Elasticsearch search parameters.
    
    Args:
        understood_query: Output from understand_query()
        
    Returns:
        Enhanced search parameters for Elasticsearch
    """
    params = {
        "query": understood_query["elasticsearch_query"],
        "boost_genres": understood_query["genres"],
        "bpm_range": understood_query["bpm_range"],
        "size": 50  # Get more candidates
    }
    
    return params


# Test function
def test_conversational_search():
    """Test various conversational queries"""
    
    test_queries = [
        "lo-fi beats for studying",
        "upbeat workout music",
        "relaxing jazz for dinner",
        "electronic music for party",
        "calm music for sleeping",
        "hip hop",  # Simple keyword
        "something to help me focus",
        "energetic rock for running"
    ]
    
    print("=" * 60)
    print("🤖 Conversational Search Test")
    print("=" * 60)
    print()
    
    for query in test_queries:
        result = understand_query(query)
        print(f"Query: \"{query}\"")
        print(f"  Mood: {result['mood']}")
        print(f"  Activity: {result['activity']}")
        print(f"  Genres: {result['genres']}")
        print(f"  BPM Range: {result['bpm_range']}")
        print(f"  Enhanced Query: \"{result['elasticsearch_query']}\"")
        print()


if __name__ == "__main__":
    test_conversational_search()
