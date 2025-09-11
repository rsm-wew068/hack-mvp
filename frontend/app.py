"""Streamlit frontend for the Music AI Recommendation System."""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🎵 AI Music Recommendations",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None


def main():
    """Main application function."""
    st.title("🎵 AI Music Recommendation System")
    st.markdown("*Powered by GPT-OSS-20B and Neural Networks*")
    
    # Sidebar for authentication
    with st.sidebar:
        st.header("🔐 Authentication")
        handle_authentication()
    
    # Main tabs
    if st.session_state.authenticated:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎵 Recommendations", 
            "🎯 Training", 
            "👤 Profile", 
            "📊 Analytics"
        ])
        
        with tab1:
            show_recommendations_tab()
        
        with tab2:
            show_training_tab()
        
        with tab3:
            show_profile_tab()
        
        with tab4:
            show_analytics_tab()
    else:
        st.info("👆 Please authenticate with Spotify in the sidebar to get started!")


def handle_authentication():
    """Handle Spotify authentication in sidebar."""
    if not st.session_state.authenticated:
        st.markdown("Connect your Spotify account to get personalized AI recommendations!")
        
        if st.button("🎵 Login to Spotify", type="primary", use_container_width=True):
            try:
                # Get auth URL from backend
                response = requests.get(f"{API_BASE_URL}/api/spotify/auth-url/demo_user")
                if response.status_code == 200:
                    auth_url = response.json()["auth_url"]
                    st.markdown(f"[Click here to authenticate with Spotify]({auth_url})")
                    st.session_state.auth_url = auth_url
                else:
                    st.error("Failed to get authentication URL")
            except Exception as e:
                st.error(f"Authentication error: {e}")
        
        # Handle callback (simplified for demo)
        if st.button("🔧 Demo Mode (Skip Auth)", type="secondary"):
            st.session_state.authenticated = True
            st.session_state.user_id = "demo_user"
            st.session_state.user_info = {
                "display_name": "Demo User",
                "id": "demo_user"
            }
            st.rerun()
    else:
        user_info = st.session_state.user_info
        st.success(f"✅ Logged in as: {user_info.get('display_name', 'User')}")
        
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.user_info = None
            st.rerun()


def show_recommendations_tab():
    """Show the recommendations tab."""
    st.header("🎯 AI-Powered Recommendations")
    
    # Get user playlists
    try:
        playlists_response = requests.get(f"{API_BASE_URL}/api/spotify/playlists/{st.session_state.user_id}")
        if playlists_response.status_code == 200:
            playlists = playlists_response.json()["playlists"]
        else:
            playlists = []
    except:
        playlists = []
    
    # Playlist selector
    if playlists:
        playlist_options = {f"{p['name']} ({p['tracks_count']} tracks)": p['id'] for p in playlists}
        selected_playlist_name = st.selectbox(
            "Choose a playlist for recommendations:",
            options=list(playlist_options.keys()),
            help="Select a playlist to get context-based recommendations"
        )
        selected_playlist_id = playlist_options[selected_playlist_name]
    else:
        st.info("No playlists found. Using general recommendations.")
        selected_playlist_id = None
    
    # Recommendation parameters
    col1, col2 = st.columns(2)
    with col1:
        num_recs = st.slider("Number of recommendations:", 1, 20, 10)
    with col2:
        confidence_threshold = st.slider("Confidence threshold:", 0.0, 1.0, 0.5)
    
    if st.button("🚀 Get AI Recommendations", type="primary", use_container_width=True):
        with st.spinner("🧠 AI is analyzing your music taste..."):
            try:
                recommendations = get_recommendations(
                    user_id=st.session_state.user_id,
                    playlist_id=selected_playlist_id,
                    num_recommendations=num_recs,
                    confidence_threshold=confidence_threshold
                )
                
                if recommendations:
                    display_recommendations(recommendations)
                else:
                    st.warning("No recommendations found. Try adjusting the confidence threshold.")
                    
            except Exception as e:
                st.error(f"Failed to get recommendations: {e}")


def display_recommendations(recommendations: List[Dict[str, Any]]):
    """Display recommendations in a nice format."""
    st.subheader("🎵 Your Personalized Recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # Album art or placeholder
                if rec.get('album_art'):
                    st.image(rec['album_art'], width=100)
                else:
                    st.image("https://via.placeholder.com/100x100/1DB954/FFFFFF?text=🎵", width=100)
            
            with col2:
                st.markdown(f"**{rec.get('name', 'Unknown Track')}**")
                st.markdown(f"*by {rec.get('artist', 'Unknown Artist')}*")
                
                # AI explanation
                if rec.get('explanation'):
                    st.markdown(f"🤖 **AI Explanation:** {rec['explanation']}")
                
                # Audio features
                if rec.get('audio_features'):
                    features = rec['audio_features']
                    st.markdown(
                        f"🎵 Energy: {features.get('energy', 0):.2f} | "
                        f"Danceability: {features.get('danceability', 0):.2f} | "
                        f"Valence: {features.get('valence', 0):.2f}"
                    )
            
            with col3:
                # Confidence score
                confidence = rec.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.2f}")
                
                # Play button
                if rec.get('spotify_url'):
                    st.markdown(f"[▶️ Open in Spotify]({rec['spotify_url']})")
                elif rec.get('preview_url'):
                    st.audio(rec['preview_url'])
            
            st.divider()


def show_training_tab():
    """Show the training tab for RLHF."""
    st.header("🎯 Train Your AI")
    st.markdown("Help the AI learn your preferences by comparing tracks!")
    
    # Get comparison tracks
    if st.button("🎲 Get New Comparison", type="primary", use_container_width=True):
        try:
            with st.spinner("Finding tracks for comparison..."):
                comparison_response = requests.get(
                    f"{API_BASE_URL}/api/training/comparison-tracks/{st.session_state.user_id}"
                )
                
                if comparison_response.status_code == 200:
                    comparison_data = comparison_response.json()
                    st.session_state.comparison_tracks = comparison_data
                    st.rerun()
                else:
                    st.error("Failed to get comparison tracks")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display comparison
    if 'comparison_tracks' in st.session_state:
        tracks = st.session_state.comparison_tracks
        track_a = tracks['track_a']
        track_b = tracks['track_b']
        
        st.subheader("Which track do you prefer?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Track A")
            if track_a.get('album_art'):
                st.image(track_a['album_art'], width=200)
            else:
                st.image("https://via.placeholder.com/200x200/1DB954/FFFFFF?text=🎵", width=200)
            
            st.markdown(f"**{track_a.get('name', 'Unknown')}**")
            st.markdown(f"*by {track_a.get('artist', 'Unknown')}*")
            
            if st.button("👍 I prefer Track A", type="primary", key="prefer_a", use_container_width=True):
                record_preference(
                    st.session_state.user_id,
                    track_a['id'],
                    track_b['id'],
                    preference=1
                )
                st.success("✅ Preference recorded! AI is learning...")
                del st.session_state.comparison_tracks
                st.rerun()
        
        with col2:
            st.markdown("### Track B")
            if track_b.get('album_art'):
                st.image(track_b['album_art'], width=200)
            else:
                st.image("https://via.placeholder.com/200x200/1DB954/FFFFFF?text=🎵", width=200)
            
            st.markdown(f"**{track_b.get('name', 'Unknown')}**")
            st.markdown(f"*by {track_b.get('artist', 'Unknown')}*")
            
            if st.button("👍 I prefer Track B", type="primary", key="prefer_b", use_container_width=True):
                record_preference(
                    st.session_state.user_id,
                    track_a['id'],
                    track_b['id'],
                    preference=0
                )
                st.success("✅ Preference recorded! AI is learning...")
                del st.session_state.comparison_tracks
                st.rerun()


def show_profile_tab():
    """Show the user profile tab."""
    st.header("👤 Your AI Profile")
    
    try:
        profile = get_user_profile(st.session_state.user_id)
        
        if profile:
            # Profile stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Sessions", profile.get('training_sessions', 0))
            with col2:
                st.metric("Preferences Recorded", profile.get('preferences_recorded', 0))
            with col3:
                st.metric("AI Confidence", f"{profile.get('ai_confidence', 0):.2f}")
            with col4:
                st.metric("Recommendations Given", profile.get('recommendations_given', 0))
            
            # Learned preferences radar chart
            st.subheader("🧠 What the AI Learned About You")
            
            preferences = profile.get('learned_preferences', {})
            if preferences:
                feature_names = list(preferences.keys())
                feature_values = list(preferences.values())
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=feature_values,
                    theta=feature_names,
                    fill='toself',
                    name='Your Musical Taste'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=False,
                    title="Your Musical Preference Profile"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Training history
            training_history = profile.get('training_history', [])
            if training_history:
                st.subheader("📈 Training History")
                df = pd.DataFrame(training_history)
                st.line_chart(df.set_index('date')['accuracy'])
        else:
            st.info("No profile data available yet. Start training the AI to build your profile!")
            
    except Exception as e:
        st.error(f"Failed to load profile: {e}")


def show_analytics_tab():
    """Show analytics and insights."""
    st.header("📊 Analytics & Insights")
    
    # Mock analytics data
    st.subheader("🎯 Recommendation Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Confidence", "0.78", "0.05")
        st.metric("User Satisfaction", "4.2/5", "0.3")
    
    with col2:
        st.metric("Recommendation Accuracy", "84%", "7%")
        st.metric("Training Progress", "67%", "12%")
    
    # Performance over time
    st.subheader("📈 Performance Over Time")
    
    # Mock performance data
    performance_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'Accuracy': [0.65 + 0.2 * (i/30) + 0.05 * (i % 7) / 7 for i in range(30)],
        'Confidence': [0.7 + 0.15 * (i/30) + 0.03 * (i % 5) / 5 for i in range(30)]
    })
    
    st.line_chart(performance_data.set_index('Date'))
    
    # Model insights
    st.subheader("🧠 AI Model Insights")
    
    insights = [
        "🎵 Your taste leans towards high-energy pop and electronic music",
        "🎸 You prefer tracks with strong vocal presence and catchy melodies",
        "🎶 The AI has learned your preference for modern production styles",
        "🎯 Your recommendations are getting more accurate with each training session"
    ]
    
    for insight in insights:
        st.info(insight)


# API helper functions
def get_recommendations(user_id: str, playlist_id: Optional[str], 
                       num_recommendations: int, confidence_threshold: float) -> List[Dict[str, Any]]:
    """Get recommendations from the API."""
    response = requests.post(f"{API_BASE_URL}/api/recommendations", json={
        "user_id": user_id,
        "playlist_id": playlist_id,
        "num_recommendations": num_recommendations,
        "confidence_threshold": confidence_threshold
    })
    
    if response.status_code == 200:
        return response.json()["recommendations"]
    else:
        raise Exception(f"API error: {response.status_code}")


def record_preference(user_id: str, track_a: str, track_b: str, preference: int):
    """Record user preference via API."""
    response = requests.post(f"{API_BASE_URL}/api/preference-feedback", json={
        "user_id": user_id,
        "track_a": track_a,
        "track_b": track_b,
        "preference": preference
    })
    
    if response.status_code != 200:
        raise Exception(f"Failed to record preference: {response.status_code}")


def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Get user profile from API."""
    response = requests.get(f"{API_BASE_URL}/api/user-profile/{user_id}")
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get profile: {response.status_code}")


if __name__ == "__main__":
    main()
