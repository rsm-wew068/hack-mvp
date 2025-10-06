-- Update BigQuery Schema to support FMA + YouTube integration
-- Adds fields for FMA data while preserving YouTube API metadata columns

-- Add missing fields to tracks table (keeping existing columns)
-- These support both FMA tracks and future YouTube video metadata
ALTER TABLE `music_ai.tracks`
ADD COLUMN IF NOT EXISTS album STRING,           -- Album name (FMA)
ADD COLUMN IF NOT EXISTS genre STRING,           -- Genre (FMA)
ADD COLUMN IF NOT EXISTS bpm FLOAT64,            -- BPM from FMA
ADD COLUMN IF NOT EXISTS key STRING,             -- Musical key (FMA)
ADD COLUMN IF NOT EXISTS duration_seconds INT64, -- Track duration
ADD COLUMN IF NOT EXISTS view_count INT64,       -- YouTube views (future)
ADD COLUMN IF NOT EXISTS like_count INT64,       -- YouTube likes (future)
ADD COLUMN IF NOT EXISTS thumbnail_url STRING,   -- YouTube thumbnail (future)
ADD COLUMN IF NOT EXISTS video_url STRING,       -- YouTube video URL (future)
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;   -- Ingestion timestamp

-- Add missing fields to audio_features table  
ALTER TABLE `music_ai.audio_features`
ADD COLUMN IF NOT EXISTS energy FLOAT64,         -- Energy level (0-1)
ADD COLUMN IF NOT EXISTS danceability FLOAT64,   -- Danceability (0-1)
ADD COLUMN IF NOT EXISTS valence FLOAT64,        -- Musical positivity (0-1)
ADD COLUMN IF NOT EXISTS acousticness FLOAT64,   -- Acoustic quality (future)
ADD COLUMN IF NOT EXISTS instrumentalness FLOAT64, -- Instrumental vs vocal (future)
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;   -- Ingestion timestamp
