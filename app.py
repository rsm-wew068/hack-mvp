"""
AI Music Discovery Engine - Streamlit Frontend
Hackathon MVP with Elastic Hybrid Search + Google Cloud AI
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
st.set_page_config(
    page_title="AI Music Discovery Engine",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1DB954;
        text-align: center;
        padding: 1rem 0;
    }
    .track-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .track-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1DB954;
        margin-bottom: 0.5rem;
    }
    .track-artist {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .track-features {
        font-size: 1rem;
        color: #888;
    }
    .score-badge {
        background-color: #1DB954;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1ed760;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{hash(str(pd.Timestamp.now()))}"
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'feedback_count' not in st.session_state:
    st.session_state.feedback_count = 0


def check_api_health() -> bool:
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def search_by_text(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Search for music using natural language"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/text-to-playlist",
            json={
                "query": query,
                "user_id": st.session_state.user_id,
                "top_k": top_k
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return None


def get_recommendations(track_id: str, top_k: int = 5) -> Dict[str, Any]:
    """Get recommendations for a seed track"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/recommend",
            json={
                "seed_track_id": track_id,
                "user_id": st.session_state.user_id,
                "top_k": top_k
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Recommendation failed: {str(e)}")
        return None


def submit_feedback(track_id: str, feedback_type: str):
    """Submit user feedback"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/feedback",
            json={
                "user_id": st.session_state.user_id,
                "track_id": track_id,
                "feedback_type": feedback_type
            },
            timeout=5
        )
        if response.status_code == 200:
            st.session_state.feedback_count += 1
            return True
        return False
    except:
        return False


def visualize_audio_features(track: Dict[str, Any]):
    """Create a radar chart for audio features"""
    # Extract features
    bpm = track.get('bpm', 0)
    key = track.get('key', 'Unknown')
    
    # Normalize BPM to 0-100 scale for visualization
    normalized_bpm = min((bpm / 200) * 100, 100) if bpm else 0
    
    # Create a simple feature visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['BPM (normalized)'],
        y=[normalized_bpm],
        marker_color='#1DB954',
        text=[f'{bpm:.0f} BPM'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f"Audio Features - {key}",
        yaxis_title="Normalized Value",
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig


def display_track_card(track: Dict[str, Any], score: float = None, explanation: str = None, index: int = 0):
    """Display a track card with details and feedback buttons"""
    track_id = track.get('track_id', 'unknown')
    title = track.get('title', 'Unknown Title')
    artist = track.get('artist', 'Unknown Artist')
    bpm = track.get('bpm', 0)
    key = track.get('key', 'Unknown')
    youtube_data = track.get('youtube_data', {})
    rlhf_boost = track.get('rlhf_boost', 0.0)
    
    with st.container():
        # YouTube video thumbnail and player
        if youtube_data:
            col_thumb, col_info = st.columns([1, 2])
            
            with col_thumb:
                thumbnail = youtube_data.get('thumbnail_medium', '')
                if thumbnail:
                    st.image(thumbnail, use_container_width=True)
                    
                # YouTube link
                watch_url = youtube_data.get('watch_url', '')
                if watch_url:
                    st.markdown(f"[▶️ Watch on YouTube]({watch_url})")
            
            with col_info:
                st.markdown(f"### 🎵 {title}")
                st.markdown(f"**Artist:** {artist}")
                st.markdown(f"**Track ID:** `{track_id}` 📋 ← *Copy this for Track Recommendations tab!*")
                if bpm:
                    st.markdown(f"**BPM:** {bpm} | **Key:** {key}")
                
                # YouTube stats
                view_count = youtube_data.get('view_count', 0)
                if view_count:
                    st.markdown(f'<div class="track-features">👁️ {view_count:,} views</div>', unsafe_allow_html=True)
        else:
            # No YouTube data - original display
            st.markdown(f'<div class="track-title">🎵 {title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="track-artist">👤 {artist}</div>', unsafe_allow_html=True)
            st.markdown(f"**Track ID:** `{track_id}` 📋")
            st.markdown(f'<div class="track-features">🎹 {key} • 🥁 {bpm:.0f} BPM</div>', unsafe_allow_html=True)
        
        # AI Explanation
        if explanation:
            st.markdown(f'<div class="track-features">💡 {explanation}</div>', unsafe_allow_html=True)
        
        # RLHF boost indicator
        if rlhf_boost != 0.0:
            boost_emoji = "⬆️" if rlhf_boost > 0 else "⬇️"
            st.markdown(
                f'<div class="track-features">{boost_emoji} RLHF Boost: {rlhf_boost:+.3f}</div>',
                unsafe_allow_html=True
            )
        
        # Score badge
        if score is not None:
            st.markdown(f'<div class="score-badge">Match Score: {score:.2%}</div>', unsafe_allow_html=True)
        
        # Action buttons row
        col1, col2, col3, col4 = st.columns(4)
        
        # Find Similar button
        with col1:
            if st.button("🔍 Find Similar", key=f"similar_{track_id}_{index}", use_container_width=True):
                st.session_state[f'show_similar_{track_id}_{index}'] = True
                st.rerun()
        
        # Feedback buttons
        with col2:
            if st.button("👍 Like", key=f"like_{track_id}_{index}", use_container_width=True):
                if submit_feedback(track_id, "like"):
                    st.success("Liked!")
                    st.rerun()
        with col3:
            if st.button("👎 Dislike", key=f"dislike_{track_id}_{index}", use_container_width=True):
                if submit_feedback(track_id, "dislike"):
                    st.warning("Disliked")
                    st.rerun()
        with col4:
            if st.button("⏭️ Skip", key=f"skip_{track_id}_{index}", use_container_width=True):
                if submit_feedback(track_id, "skip"):
                    st.info("Skipped")
                    st.rerun()
        
        # Expandable sections
        col_features, col_video = st.columns(2)
        
        with col_features:
            with st.expander("📊 Audio Features"):
                fig = visualize_audio_features(track)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{track_id}_{index}")
        
        with col_video:
            if youtube_data and youtube_data.get('embed_url'):
                with st.expander("📺 YouTube Video"):
                    st.components.v1.iframe(youtube_data['embed_url'], height=200)
        
        # Show similar tracks if button was clicked
        if st.session_state.get(f'show_similar_{track_id}_{index}', False):
            with st.expander(f"🔍 Similar to '{title}'", expanded=True):
                similar_results = get_recommendations(track_id, top_k=5)
                if similar_results and similar_results.get('recommendations'):
                    st.success(f"✅ Found {len(similar_results['recommendations'])} similar tracks!")
                    for j, rec in enumerate(similar_results['recommendations']):
                        display_track_card(
                            rec['track'],
                            rec.get('score'),
                            rec.get('explanation'),
                            index=index*1000+j  # Unique index to avoid conflicts
                        )
                        st.markdown("---")
                else:
                    st.error("Could not find similar tracks")
                
                # Reset button to close
                if st.button("✖️ Close", key=f"close_similar_{track_id}_{index}"):
                    st.session_state[f'show_similar_{track_id}_{index}'] = False
                    st.rerun()
        
        st.markdown("---")
        
        st.markdown("---")


def main():
    # Header
    st.markdown('<div class="main-header">🎵 AI Music Discovery Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #666;">Powered by Elastic Hybrid Search + Google Cloud AI</p>',
        unsafe_allow_html=True
    )
    
    # Check API health
    api_status = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Status
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Offline")
            st.warning(f"Make sure the API is running at {API_BASE_URL}")
            st.code("python main.py", language="bash")
        
        st.markdown("---")
        
        # User info
        st.subheader("👤 User Info")
        st.text(f"ID: {st.session_state.user_id[:12]}...")
        st.text(f"Feedback: {st.session_state.feedback_count}")
        
        st.markdown("---")
        
        # Search settings
        st.subheader("🎛️ Search Settings")
        top_k = st.slider("Results to show", 1, 10, 5)
        
        st.markdown("---")
        
        # Search history
        if st.session_state.search_history:
            st.subheader("📜 Recent Searches")
            for i, query in enumerate(reversed(st.session_state.search_history[-5:])):
                st.text(f"{i+1}. {query[:30]}...")
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        This demo uses:
        - **Elasticsearch** for hybrid search (BM25 + k-NN)
        - **BigQuery** for track metadata
        - **OpenL3** embeddings for audio similarity
        - **FastAPI** backend
        - **Streamlit** frontend
        """)
    
    # Main content
    if not api_status:
        st.error("⚠️ Cannot connect to API. Please start the backend server first.")
        st.code("python main.py", language="bash")
        return
    
    # Search tabs
    tab1, tab2 = st.tabs(["🔍 Text Search", "🎯 Track Recommendations"])
    
    with tab1:
        st.subheader("🤖 Conversational Music Search")
        st.markdown("*Powered by AI mood understanding - Just describe what you want!*")
        
        # Example queries in an expander
        with st.expander("💡 Try these natural language queries", expanded=False):
            st.markdown("""
            **Study & Focus:**
            - *"lo-fi beats for studying"*
            - *"something to help me focus"*
            - *"calm music for concentration"*
            
            **Energy & Workout:**
            - *"upbeat workout music"*
            - *"energetic rock for running"*
            - *"intense electronic pump music"*
            
            **Relax & Chill:**
            - *"relaxing jazz for dinner"*
            - *"calm music for sleeping"*
            - *"peaceful ambient sounds"*
            
            **Party & Social:**
            - *"electronic music for party"*
            - *"dance club beats"*
            - *"groovy music for driving"*
            
            **Or just search by genre:** *"hip hop"*, *"classical"*, *"rock"*
            """)
            
            # Quick example buttons
            st.markdown("**Quick examples (click to try):**")
            example_queries = [
                "lo-fi beats for studying 📚",
                "upbeat workout music 💪",
                "relaxing jazz for dinner 🍷",
                "electronic dance party music 🎉",
            ]
            cols = st.columns(4)
            for idx, example in enumerate(example_queries):
                with cols[idx]:
                    # Extract just the text without emoji for the query
                    query_text = example.rsplit(" ", 1)[0]  # Remove last word (emoji)
                    if st.button(example, key=f"example_{idx}", use_container_width=True):
                        st.session_state["current_search_query"] = query_text
                        st.rerun()
        
        # Search input
        col1, col2 = st.columns([4, 1])
        with col1:
            # Use session state to populate from example buttons
            default_query = st.session_state.get("current_search_query", "")
            search_query = st.text_input(
                "What kind of music are you looking for?",
                value=default_query,
                placeholder="e.g., 'relaxing piano music for studying' or 'energetic rock'",
                label_visibility="collapsed"
            )
            # Clear session state after using it
            if "current_search_query" in st.session_state:
                del st.session_state["current_search_query"]
        with col2:
            search_button = st.button("🔍 Search", use_container_width=True)
        
        # Perform search
        if search_button and search_query:
            with st.spinner("🎵 Searching for music..."):
                results = search_by_text(search_query, top_k)
                
                if results and results.get('recommendations'):
                    # Save to history
                    st.session_state.search_history.append(search_query)
                    
                    st.success(f"✅ Found {results.get('total_candidates', 0)} matches!")
                    st.markdown("---")
                    
                    # Display results
                    for i, rec in enumerate(results['recommendations']):
                        display_track_card(
                            rec['track'],
                            rec.get('score'),
                            rec.get('explanation'),
                            index=i
                        )
                else:
                    st.warning("No results found. Try a different search query.")
    
    with tab2:
        st.subheader("🎯 Get Recommendations Based on a Track")
        st.markdown("*Find similar tracks based on audio similarity*")
        
        # Prominent instructions
        st.info("""
        📋 **Need a Track ID?** 
        1. Go to **Text Search** tab above
        2. Search for any music (e.g., "hip hop", "rock", "electronic")
        3. **Copy the Track ID** (looks like `fma_000123` with 📋 icon)
        4. Come back here and paste it below!
        """)
        
        # Simple track ID input
        col1, col2 = st.columns([4, 1])
        with col1:
            track_input = st.text_input(
                "Enter Track ID:",
                placeholder="e.g., fma_000001",
                key="track_id_input"
            )
        with col2:
            recommend_button = st.button("🎯 Get Similar", key="get_similar_btn")
        
        # Show recommendations
        if recommend_button and track_input:
            with st.spinner(f"🎵 Finding similar tracks..."):
                results = get_recommendations(track_input, top_k)
                
                if results and results.get('recommendations'):
                    st.success(f"✅ Found {len(results['recommendations'])} similar tracks!")
                    st.markdown("---")
                    
                    # Display recommendations
                    for i, rec in enumerate(results['recommendations']):
                        display_track_card(
                            rec['track'],
                            rec.get('score'),
                            rec.get('explanation'),
                            index=i+100  # Offset to avoid key conflicts
                        )
                else:
                    st.error("Could not find that track. Try a different track ID.")
        elif recommend_button:
            st.warning("⚠️ Please enter a track ID first! Search in Text Search tab to find track IDs.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888; font-size: 0.9rem;">Built for Google Cloud + Elastic Hackathon 2025 🚀</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
