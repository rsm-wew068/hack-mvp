# Cloud Run FastAPI Service
# AI Music Discovery Engine - Google Cloud + Elastic

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import logging
from google.cloud import bigquery
from elasticsearch import Elasticsearch
from cloud_run_scoring import recommend_tracks, calculate_recommendation_score
from youtube_integration import YouTubeClient, enrich_track_with_youtube
from rlhf_reranker import RLHFReranker
from conversational_search import understand_query, enhance_search_params
from ai_explainer import get_explainer, generate_explanation, generate_playlist_explanation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Music Discovery Engine",
    description="Conversational music discovery with Elastic hybrid search and Google Cloud AI",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize clients
bigquery_client = None
elastic_client = None
youtube_client = None
rlhf_reranker = None
ai_explainer = None

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global bigquery_client, elastic_client, youtube_client, rlhf_reranker
    
    # Initialize BigQuery client
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if project_id:
        try:
            bigquery_client = bigquery.Client(project=project_id)
            print(f"✅ Connected to BigQuery project: {project_id}")
        except Exception as e:
            print(f"⚠️  BigQuery initialization failed: {e}")
            bigquery_client = None
    else:
        print("⚠️  GOOGLE_CLOUD_PROJECT not set")
    
    # Initialize Elastic client
    es_url = os.getenv('ELASTICSEARCH_URL')
    api_key = os.getenv('ELASTIC_API_KEY')
    
    if es_url and api_key:
        try:
            elastic_client = Elasticsearch(
                hosts=[es_url],
                api_key=api_key
            )
            if elastic_client.ping():
                print(f"✅ Connected to Elasticsearch at {es_url}")
            else:
                print("❌ Failed to connect to Elasticsearch")
                elastic_client = None
        except Exception as e:
            print(f"⚠️  Elasticsearch initialization failed: {e}")
            elastic_client = None
    else:
        print("⚠️  Elastic credentials not set")
    
    # Initialize YouTube client
    try:
        youtube_client = YouTubeClient()
        print("✅ YouTube client initialized")
    except Exception as e:
        print(f"⚠️  YouTube initialization failed: {e}")
        youtube_client = None
    
    # Initialize RLHF reranker
    try:
        rlhf_reranker = RLHFReranker(project_id=project_id)
        print("✅ RLHF reranker initialized")
    except Exception as e:
        print(f"⚠️  RLHF initialization failed: {e}")
        rlhf_reranker = None
    
    # Initialize AI Explainer (Vertex AI Gemini)
    try:
        ai_explainer = get_explainer()
        print("✅ AI Explainer (Vertex AI Gemini) initialized")
    except Exception as e:
        print(f"⚠️  AI Explainer initialization failed: {e}")
        ai_explainer = None


# Root endpoint - serve the frontend
@app.get("/")
async def root():
    """Serve the main HTML interface"""
    return FileResponse("static/index.html")


# Request/Response Models
class RecommendationRequest(BaseModel):
    query: Optional[str] = None
    seed_track_id: Optional[str] = None
    user_id: Optional[str] = None
    top_k: int = 10

class Track(BaseModel):
    track_id: str
    title: str
    artist: str
    bpm: Optional[float] = None
    key: Optional[str] = None
    openl3: List[float]
    yt_video_id: Optional[str] = None

class Recommendation(BaseModel):
    track: Track
    score: float
    explanation: str

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]
    query: Optional[str]
    seed_track_id: Optional[str]
    total_candidates: int
    explanation: Optional[str] = None  # AI-generated playlist explanation

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Music Discovery Engine"}

# Main recommendation endpoint
@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get AI-powered music recommendations using Elastic hybrid search.
    
    Supports both text queries and seed track similarity.
    """
    try:
        # Get candidates from Elastic (placeholder - implement Elastic client)
        candidates = await get_elastic_candidates(request.query, request.seed_track_id)
        
        if not candidates:
            raise HTTPException(status_code=404, detail="No candidates found")
        
        # Get seed track for similarity scoring
        seed_track = await get_seed_track(request.seed_track_id, request.query)
        
        # Get user profile for personalization
        user_profile = await get_user_profile(request.user_id) if request.user_id else None
        
        # Generate recommendations with scoring
        recommendations = recommend_tracks(
            seed_track=seed_track,
            candidates=candidates,
            user_profile=user_profile,
            top_k=request.top_k * 2  # Get more for reranking
        )
        
        # Apply RLHF reranking if user_id provided
        if request.user_id and rlhf_reranker:
            recommendations = rlhf_reranker.rerank_recommendations(
                recommendations,
                request.user_id,
                max_boost=0.2
            )
        
        # Take top_k after reranking
        recommendations = recommendations[:request.top_k]
        
        # 🆕 Generate AI explanations for each recommendation using Vertex AI Gemini
        if ai_explainer and seed_track:
            for rec in recommendations:
                track = rec.get('track', {})
                try:
                    # Generate AI explanation using Vertex AI
                    audio_features = {
                        'bpm_diff': abs(track.get('bpm', 120) - seed_track.get('bpm', 120)),
                        'key_match': 'same' if track.get('key') == seed_track.get('key') else 'different',
                        'genre': track.get('genre', 'Unknown')
                    }
                    
                    explanation = generate_explanation(
                        seed_track=seed_track,
                        recommended_track=track,
                        similarity_score=rec.get('score', 0.0),
                        audio_features=audio_features
                    )
                    
                    rec['explanation'] = explanation
                    logger.info(f"✨ AI Explanation: {explanation}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to generate explanation: {e}")
                    rec['explanation'] = f"{rec.get('score', 0):.0%} match"
        
        # Enrich with YouTube metadata
        if youtube_client and youtube_client.youtube:
            for rec in recommendations:
                track = rec.get('track', {})
                if track.get('title') and track.get('artist'):
                    video_data = youtube_client.search_music_video(
                        title=track['title'],
                        artist=track['artist']
                    )
                    if video_data:
                        track['youtube_data'] = video_data
                        track['yt_video_id'] = video_data['video_id']
        
        return RecommendationResponse(
            recommendations=recommendations,
            query=request.query,
            seed_track_id=request.seed_track_id,
            total_candidates=len(candidates)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text-to-playlist endpoint
@app.post("/api/text-to-playlist", response_model=RecommendationResponse)
async def text_to_playlist(request: RecommendationRequest):
    """
    Generate playlist from natural language description.
    Example: "lo-fi beats for studying" -> curated playlist with mood understanding
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required for text-to-playlist")
    
    # 🆕 Understand natural language query
    understood = understand_query(request.query)
    
    # Log conversational understanding
    logger.info(f"🤖 Conversational Query: {request.query}")
    logger.info(f"   → Mood: {understood['mood']}, Activity: {understood['activity']}")
    logger.info(f"   → Genres: {understood['genres']}")
    logger.info(f"   → BPM Range: {understood['bpm_range']}")
    
    # Create enhanced request with mood-aware query
    enhanced_request = RecommendationRequest(
        query=understood["elasticsearch_query"],  # Enhanced query
        seed_track_id=request.seed_track_id,
        user_id=request.user_id,
        top_k=request.top_k
    )
    
    # Get recommendations with enhanced query
    response = await get_recommendations(enhanced_request)
    
    # 🆕 Add AI-powered playlist explanation using Vertex AI Gemini
    if ai_explainer and response.recommendations:
        try:
            tracks = [rec.get('track', {}) for rec in response.recommendations]
            playlist_explanation = generate_playlist_explanation(
                query=request.query,
                tracks=tracks,
                mood=understood.get('mood')
            )
            # Add to metadata or first track explanation
            logger.info(f"🎵 Playlist Explanation: {playlist_explanation}")
            response.explanation = playlist_explanation  # Add to response if model supports
        except Exception as e:
            logger.warning(f"⚠️  Failed to generate playlist explanation: {e}")
    
    return response

# User feedback endpoint
@app.post("/api/feedback")
async def record_feedback(
    user_id: str,
    track_id: str,
    event: str  # 'view', 'click', 'like', 'skip'
):
    """Record user feedback for personalization."""
    try:
        # Store feedback in BigQuery (placeholder)
        await store_user_feedback(user_id, track_id, event)
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Real implementation with BigQuery and Elastic
async def get_elastic_candidates(
    query: Optional[str], 
    seed_track_id: Optional[str]
) -> List[Dict]:
    """Get candidates from Elastic using hybrid search."""
    if not elastic_client:
        raise HTTPException(
            status_code=503, 
            detail="Elastic service not available"
        )
    
    index_name = "music-tracks"
    candidates = []
    
    try:
        if query:
            # Hybrid search: BM25 across multiple fields including genre
            search_body = {
                "size": 50,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^3", "artist^2", "genre^4", "album"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            },
                            {
                                "match": {
                                    "genre": {
                                        "query": query,
                                        "boost": 5
                                    }
                                }
                            }
                        ]
                    }
                }
            }
            
            response = elastic_client.search(
                index=index_name, 
                body=search_body
            )
            
            for hit in response['hits']['hits']:
                candidates.append(hit['_source'])
        
        elif seed_track_id:
            # k-NN similarity search
            # First get the seed track's embedding
            seed_doc = elastic_client.get(
                index=index_name, 
                id=seed_track_id
            )
            seed_embedding = seed_doc['_source']['openl3']
            
            # k-NN search
            search_body = {
                "size": 50,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'openl3') + 1.0",
                            "params": {"query_vector": seed_embedding}
                        }
                    }
                }
            }
            
            response = elastic_client.search(
                index=index_name, 
                body=search_body
            )
            
            for hit in response['hits']['hits']:
                if hit['_id'] != seed_track_id:
                    candidates.append(hit['_source'])
        
        else:
            # Default: return some tracks
            response = elastic_client.search(
                index=index_name,
                body={"size": 50, "query": {"match_all": {}}}
            )
            
            for hit in response['hits']['hits']:
                candidates.append(hit['_source'])
        
        return candidates
    
    except Exception as e:
        print(f"Error in Elastic search: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Search error: {str(e)}"
        )


async def get_seed_track(
    seed_track_id: Optional[str], 
    query: Optional[str]
) -> Dict:
    """Get seed track for similarity scoring."""
    if seed_track_id:
        if not bigquery_client:
            raise HTTPException(
                status_code=503, 
                detail="BigQuery service not available"
            )
        
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        
        # Query BigQuery for track features
        query_sql = f"""
        SELECT 
            t.track_id,
            t.title,
            t.artist,
            af.bpm,
            af.key,
            af.openl3
        FROM `{project_id}.music_ai.tracks` t
        JOIN `{project_id}.music_ai.audio_features` af
        ON t.track_id = af.track_id
        WHERE t.track_id = @track_id
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "track_id", 
                    "STRING", 
                    seed_track_id
                )
            ]
        )
        
        query_job = bigquery_client.query(query_sql, job_config=job_config)
        results = list(query_job.result())
        
        if results:
            row = results[0]
            return {
                "track_id": row.track_id,
                "title": row.title,
                "artist": row.artist,
                "bpm": row.bpm,
                "key": row.key,
                "openl3": row.openl3
            }
    
    # Default seed based on query or first track
    if elastic_client:
        try:
            response = elastic_client.search(
                index="music-tracks",
                body={"size": 1, "query": {"match_all": {}}}
            )
            if response['hits']['hits']:
                return response['hits']['hits'][0]['_source']
        except:
            pass
    
    # Fallback
    return {
        "track_id": "default",
        "title": "Default",
        "artist": "Unknown",
        "bpm": 120.0,
        "key": "C major",
        "openl3": [0.0] * 512
    }


async def get_user_profile(user_id: str) -> Optional[Dict]:
    """Get user profile for personalization."""
    if not bigquery_client:
        return None
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    query_sql = f"""
    SELECT 
        user_id,
        preferred_openl3_centroid,
        preferred_bpm_range,
        preferred_keys,
        diversity_score
    FROM `{project_id}.music_ai.user_profiles`
    WHERE user_id = @user_id
    LIMIT 1
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
        ]
    )
    
    try:
        query_job = bigquery_client.query(query_sql, job_config=job_config)
        results = list(query_job.result())
        
        if results:
            row = results[0]
            return {
                "user_id": row.user_id,
                "preferred_openl3_centroid": row.preferred_openl3_centroid,
                "preferred_bpm_range": row.preferred_bpm_range,
                "preferred_keys": row.preferred_keys,
                "diversity_score": row.diversity_score
            }
    except Exception as e:
        print(f"Error fetching user profile: {e}")
    
    return None


async def store_user_feedback(user_id: str, track_id: str, event: str):
    """Store user feedback in BigQuery."""
    if not bigquery_client:
        return
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    table_id = f"{project_id}.music_ai.user_feedback"
    
    rows_to_insert = [{
        "user_id": user_id,
        "track_id": track_id,
        "event": event
    }]
    
    try:
        errors = bigquery_client.insert_rows_json(table_id, rows_to_insert)
        if errors:
            print(f"Error inserting feedback: {errors}")
        else:
            print(f"✅ Feedback recorded: {user_id} -> {track_id} ({event})")
    except Exception as e:
        print(f"Error storing feedback: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
