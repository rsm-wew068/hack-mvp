-- BigQuery Schema for AI Music Recommendation System
-- Elastic Challenge: AI-Powered Search with Google Cloud
-- Minimal, Shippable Specs

-- Tracks table: Core music metadata
CREATE TABLE `music_ai.tracks` (
  track_id STRING NOT NULL,
  title STRING NOT NULL,
  artist STRING NOT NULL,
  license STRING NOT NULL, -- 'cc', 'public_domain', 'user_upload'
  source_url STRING, -- Original audio source
  yt_video_id STRING -- Optional: for linking to YouTube videos
)
PARTITION BY DATE(CURRENT_TIMESTAMP())
CLUSTER BY artist;

-- Audio features table: Technical audio analysis
CREATE TABLE `music_ai.audio_features` (
  track_id STRING NOT NULL,
  bpm FLOAT64, -- Beats per minute
  key STRING, -- e.g., "A minor", "C major"
  mfcc_mean ARRAY<FLOAT64>, -- 13 MFCC coefficients
  spectral FLOAT64, -- Spectral centroid
  openl3 ARRAY<FLOAT64> -- 512-dimensional OpenL3 embedding
)
PARTITION BY DATE(CURRENT_TIMESTAMP())
CLUSTER BY track_id;

-- User feedback table: For personalization
CREATE TABLE `music_ai.user_feedback` (
  user_id STRING NOT NULL,
  track_id STRING NOT NULL,
  event STRING NOT NULL, -- 'view', 'click', 'like', 'skip'
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(ts)
CLUSTER BY user_id, track_id;

-- User profiles table: Aggregated user preferences
CREATE TABLE `music_ai.user_profiles` (
  user_id STRING NOT NULL,
  
  -- Audio preference vectors
  preferred_openl3_centroid ARRAY<FLOAT64>,
  preferred_bpm_range STRUCT<min_bpm FLOAT64, max_bpm FLOAT64, avg_bpm FLOAT64>,
  preferred_keys ARRAY<STRING>, -- Most liked keys
  preferred_genres ARRAY<STRING>,
  
  -- Behavioral patterns
  listening_sessions_count INT64,
  total_listening_time_seconds INT64,
  diversity_score FLOAT64, -- How diverse their taste is
  
  -- RLHF feedback
  feedback_count INT64,
  positive_feedback_ratio FLOAT64,
  
  -- Profile metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  profile_version STRING DEFAULT 'v1.0'
)
PARTITION BY DATE(last_updated)
CLUSTER BY user_id;

-- Recommendations table: Generated recommendations with explanations
CREATE TABLE `music_ai.recommendations` (
  recommendation_id STRING NOT NULL,
  user_id STRING NOT NULL,
  track_id STRING NOT NULL,
  
  -- Recommendation scores
  overall_score FLOAT64 NOT NULL, -- 0.0 to 1.0
  audio_similarity_score FLOAT64,
  bpm_match_score FLOAT64,
  key_match_score FLOAT64,
  genre_match_score FLOAT64,
  
  -- Explanation components
  explanation_text STRING, -- AI-generated explanation
  rationale_chips ARRAY<STRING>, -- e.g., ["82 BPM", "A minor", "timbre: 0.18"]
  
  -- Recommendation metadata
  recommendation_type STRING, -- 'collaborative', 'content_based', 'hybrid'
  novelty_score FLOAT64, -- How novel this recommendation is
  confidence_score FLOAT64, -- AI confidence in recommendation
  
  -- Generation metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  model_version STRING DEFAULT 'v1.0',
  generation_time_ms INT64
)
PARTITION BY DATE(created_at)
CLUSTER BY user_id, track_id;

-- Search queries table: For analytics and improvement
CREATE TABLE `music_ai.search_queries` (
  query_id STRING NOT NULL,
  user_id STRING,
  query_text STRING,
  query_type STRING, -- 'text', 'audio', 'hybrid'
  
  -- Query results
  results_count INT64,
  clicked_track_id STRING,
  click_position INT64,
  
  -- Query metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  session_id STRING,
  user_agent STRING
)
PARTITION BY DATE(created_at)
CLUSTER BY user_id, query_text;

-- Index for fast similarity search
CREATE TABLE `music_ai.audio_embeddings_index` (
  track_id STRING NOT NULL,
  openl3_embedding ARRAY<FLOAT64> NOT NULL,
  embedding_norm FLOAT64 NOT NULL, -- Precomputed L2 norm for cosine similarity
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(created_at)
CLUSTER BY track_id;

-- Views for common queries
CREATE VIEW `music_ai.track_features_view` AS
SELECT 
  t.track_id,
  t.title,
  t.artist,
  t.genre,
  t.duration_seconds,
  af.bpm,
  af.key_signature,
  af.key_confidence,
  af.spectral_centroid,
  af.rms_energy,
  af.openl3_embedding
FROM `music_ai.tracks` t
JOIN `music_ai.audio_features` af ON t.track_id = af.track_id;

-- View for user recommendation history
CREATE VIEW `music_ai.user_recommendation_history` AS
SELECT 
  r.user_id,
  r.track_id,
  t.title,
  t.artist,
  r.overall_score,
  r.explanation_text,
  r.created_at
FROM `music_ai.recommendations` r
JOIN `music_ai.tracks` t ON r.track_id = t.track_id
ORDER BY r.user_id, r.created_at DESC;
