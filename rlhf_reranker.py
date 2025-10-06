#!/usr/bin/env python3
"""
RLHF (Reinforcement Learning from Human Feedback) Re-ranker
Adjusts recommendation scores based on user feedback patterns
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from google.cloud import bigquery
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()


class RLHFReranker:
    """Re-rank recommendations based on user feedback history"""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize RLHF reranker
        
        Args:
            project_id: Google Cloud project ID
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        try:
            self.bq_client = bigquery.Client(project=self.project_id)
            print("✅ RLHF Reranker initialized")
        except Exception as e:
            print(f"⚠️  BigQuery client init failed: {e}")
            self.bq_client = None
        
        # Feedback weights
        self.feedback_weights = {
            'like': 1.0,
            'play': 0.5,
            'click': 0.3,
            'view': 0.1,
            'skip': -0.5,
            'dislike': -1.0
        }
    
    def get_user_feedback_history(
        self,
        user_id: str,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Get user's feedback history from BigQuery
        
        Args:
            user_id: User ID
            limit: Max number of feedback records
        
        Returns:
            List of feedback records
        """
        if not self.bq_client:
            return []
        
        query = f"""
        SELECT 
            track_id,
            event,
            ts
        FROM `{self.project_id}.music_ai.user_feedback`
        WHERE user_id = @user_id
        ORDER BY ts DESC
        LIMIT {limit}
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        
        try:
            query_job = self.bq_client.query(query, job_config=job_config)
            results = list(query_job.result())
            
            return [
                {
                    'track_id': row.track_id,
                    'event': row.event,
                    'timestamp': row.ts
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error fetching feedback: {e}")
            return []
    
    def get_track_features_for_feedback(
        self,
        track_ids: List[str]
    ) -> Dict[str, Dict]:
        """
        Get audio features for tracks from feedback history
        
        Args:
            track_ids: List of track IDs
        
        Returns:
            Dict mapping track_id to features
        """
        if not self.bq_client or not track_ids:
            return {}
        
        # Create parameterized query
        track_ids_str = ', '.join([f"'{tid}'" for tid in track_ids])
        
        query = f"""
        SELECT 
            af.track_id,
            af.bpm,
            af.key,
            af.openl3,
            af.energy,
            af.danceability,
            af.valence,
            t.genre
        FROM `{self.project_id}.music_ai.audio_features` af
        JOIN `{self.project_id}.music_ai.tracks` t
        ON af.track_id = t.track_id
        WHERE af.track_id IN ({track_ids_str})
        """
        
        try:
            query_job = self.bq_client.query(query)
            results = list(query_job.result())
            
            features_map = {}
            for row in results:
                features_map[row.track_id] = {
                    'bpm': row.bpm,
                    'key': row.key,
                    'openl3': list(row.openl3) if row.openl3 else None,
                    'energy': row.energy if hasattr(row, 'energy') else None,
                    'danceability': row.danceability if hasattr(row, 'danceability') else None,
                    'valence': row.valence if hasattr(row, 'valence') else None,
                    'genre': row.genre if hasattr(row, 'genre') else None
                }
            
            return features_map
        except Exception as e:
            print(f"Error fetching track features: {e}")
            return {}
    
    def calculate_user_preference_vector(
        self,
        feedback_history: List[Dict],
        track_features: Dict[str, Dict]
    ) -> Optional[np.ndarray]:
        """
        Calculate user's preference vector from feedback
        
        Args:
            feedback_history: User's feedback records
            track_features: Track features map
        
        Returns:
            Preference vector (weighted average of liked track embeddings)
        """
        weighted_embeddings = []
        
        for feedback in feedback_history:
            track_id = feedback['track_id']
            event = feedback['event']
            
            # Get weight for this event type
            weight = self.feedback_weights.get(event, 0.0)
            
            if weight == 0.0:
                continue
            
            # Get track features
            features = track_features.get(track_id)
            if not features or not features.get('openl3'):
                continue
            
            embedding = np.array(features['openl3'])
            weighted_embeddings.append((embedding, weight))
        
        if not weighted_embeddings:
            return None
        
        # Calculate weighted centroid
        total_weight = sum(abs(w) for _, w in weighted_embeddings)
        if total_weight == 0:
            return None
        
        preference_vector = sum(
            emb * w for emb, w in weighted_embeddings
        ) / total_weight
        
        return preference_vector
    
    def calculate_feedback_boost(
        self,
        candidate_embedding: np.ndarray,
        preference_vector: np.ndarray,
        max_boost: float = 0.2
    ) -> float:
        """
        Calculate score boost based on similarity to user preferences
        
        Args:
            candidate_embedding: Candidate track embedding
            preference_vector: User preference vector
            max_boost: Maximum boost to apply
        
        Returns:
            Score adjustment (-max_boost to +max_boost)
        """
        # Calculate cosine similarity
        dot_product = np.dot(candidate_embedding, preference_vector)
        norm1 = np.linalg.norm(candidate_embedding)
        norm2 = np.linalg.norm(preference_vector)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Map similarity [-1, 1] to boost [-max_boost, +max_boost]
        boost = similarity * max_boost
        
        return boost
    
    def rerank_recommendations(
        self,
        recommendations: List[Dict],
        user_id: str,
        max_boost: float = 0.2
    ) -> List[Dict]:
        """
        Re-rank recommendations based on user feedback
        
        Args:
            recommendations: List of recommendation dicts with 'track' and 'score'
            user_id: User ID for feedback lookup
            max_boost: Maximum score adjustment
        
        Returns:
            Re-ranked recommendations with updated scores
        """
        # Get user feedback history
        feedback_history = self.get_user_feedback_history(user_id)
        
        if not feedback_history:
            print(f"No feedback history for user {user_id}")
            return recommendations
        
        print(f"Found {len(feedback_history)} feedback records for {user_id}")
        
        # Get track IDs from feedback
        feedback_track_ids = list(set(
            f['track_id'] for f in feedback_history
        ))
        
        # Get features for feedback tracks
        track_features = self.get_track_features_for_feedback(
            feedback_track_ids
        )
        
        if not track_features:
            print("Could not fetch track features")
            return recommendations
        
        # Calculate user preference vector
        preference_vector = self.calculate_user_preference_vector(
            feedback_history,
            track_features
        )
        
        if preference_vector is None:
            print("Could not calculate preference vector")
            return recommendations
        
        print(f"✅ Calculated user preference vector")
        
        # Apply RLHF boost to each recommendation
        reranked = []
        
        for rec in recommendations:
            track = rec.get('track', {})
            base_score = rec.get('score', 0.0)
            
            # Get candidate embedding
            embedding = track.get('openl3')
            
            if embedding:
                embedding_array = np.array(embedding)
                
                # Calculate boost
                boost = self.calculate_feedback_boost(
                    embedding_array,
                    preference_vector,
                    max_boost
                )
                
                # Apply boost
                new_score = base_score + boost
                new_score = max(0.0, min(1.0, new_score))  # Clamp
                
                rec['score'] = new_score
                rec['rlhf_boost'] = boost
                rec['rlhf_applied'] = True
            else:
                rec['rlhf_boost'] = 0.0
                rec['rlhf_applied'] = False
            
            reranked.append(rec)
        
        # Re-sort by new scores
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"✅ RLHF reranking complete")
        
        return reranked


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Testing RLHF Reranker")
    print("=" * 60)
    
    reranker = RLHFReranker()
    
    if reranker.bq_client:
        print("\n📊 Test: Get user feedback history")
        
        test_user_id = "test_user"
        feedback = reranker.get_user_feedback_history(test_user_id)
        
        print(f"Found {len(feedback)} feedback records")
        
        if feedback:
            print("\nSample feedback:")
            for f in feedback[:5]:
                print(f"  {f['event']}: {f['track_id']}")
        
        print("\n📊 Test: Mock reranking")
        
        # Mock recommendations
        mock_recs = [
            {
                'track': {
                    'track_id': '001',
                    'title': 'Track 1',
                    'openl3': [0.1] * 512
                },
                'score': 0.8
            },
            {
                'track': {
                    'track_id': '002',
                    'title': 'Track 2',
                    'openl3': [0.2] * 512
                },
                'score': 0.75
            }
        ]
        
        reranked = reranker.rerank_recommendations(
            mock_recs,
            test_user_id
        )
        
        print(f"\n✅ Reranked {len(reranked)} recommendations")
        
        for rec in reranked:
            boost = rec.get('rlhf_boost', 0.0)
            print(f"  {rec['track']['title']}: {rec['score']:.3f} (boost: {boost:+.3f})")
    else:
        print("\n⚠️  BigQuery not available")
    
    print("\n" + "=" * 60)
