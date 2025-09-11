-- Database schema for Music AI Recommendation System

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(255) PRIMARY KEY,
    spotify_id VARCHAR(255) UNIQUE,
    display_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tracks table
CREATE TABLE IF NOT EXISTS tracks (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    artist VARCHAR(255) NOT NULL,
    album VARCHAR(255),
    duration_ms INTEGER,
    popularity INTEGER,
    explicit BOOLEAN DEFAULT FALSE,
    audio_features JSON,
    embedding BLOB, -- Store audio embedding as binary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Playlists table
CREATE TABLE IF NOT EXISTS playlists (
    id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tracks_count INTEGER DEFAULT 0,
    public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Playlist tracks junction table
CREATE TABLE IF NOT EXISTS playlist_tracks (
    playlist_id VARCHAR(255) NOT NULL,
    track_id VARCHAR(255) NOT NULL,
    position INTEGER,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (playlist_id, track_id),
    FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- User preferences for RLHF training
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    track_a_id VARCHAR(255) NOT NULL,
    track_b_id VARCHAR(255) NOT NULL,
    preference INTEGER NOT NULL, -- 1 for track_a, 0 for track_b
    confidence REAL DEFAULT 1.0, -- User's confidence in their choice
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (track_a_id) REFERENCES tracks(id) ON DELETE CASCADE,
    FOREIGN KEY (track_b_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- Recommendations history
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    track_id VARCHAR(255) NOT NULL,
    playlist_id VARCHAR(255), -- Context playlist if applicable
    confidence_score REAL NOT NULL,
    explanation TEXT,
    source VARCHAR(50), -- 'collaborative', 'audio_similarity', 'hybrid'
    user_feedback INTEGER, -- 1 for liked, 0 for disliked, NULL for no feedback
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE SET NULL
);

-- Model checkpoints and performance metrics
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(255) NOT NULL,
    checkpoint_path VARCHAR(500) NOT NULL,
    performance_metrics JSON,
    training_samples INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions for tracking engagement
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(255) NOT NULL,
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP,
    recommendations_requested INTEGER DEFAULT 0,
    preferences_recorded INTEGER DEFAULT 0,
    tracks_played INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Audio features cache for performance
CREATE TABLE IF NOT EXISTS audio_features_cache (
    track_id VARCHAR(255) PRIMARY KEY,
    features JSON NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_spotify_id ON users(spotify_id);
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
CREATE INDEX IF NOT EXISTS idx_tracks_popularity ON tracks(popularity);
CREATE INDEX IF NOT EXISTS idx_playlists_user_id ON playlists(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_created_at ON user_preferences(created_at);
CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON recommendations(created_at);
CREATE INDEX IF NOT EXISTS idx_recommendations_confidence ON recommendations(confidence_score);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_start ON user_sessions(session_start);

-- Views for common queries
CREATE VIEW IF NOT EXISTS user_stats AS
SELECT 
    u.id,
    u.display_name,
    COUNT(DISTINCT p.id) as playlist_count,
    COUNT(DISTINCT pt.track_id) as unique_tracks_played,
    COUNT(DISTINCT up.id) as preferences_recorded,
    AVG(r.confidence_score) as avg_recommendation_confidence,
    MAX(r.created_at) as last_recommendation_at
FROM users u
LEFT JOIN playlists p ON u.id = p.user_id
LEFT JOIN playlist_tracks pt ON p.id = pt.playlist_id
LEFT JOIN user_preferences up ON u.id = up.user_id
LEFT JOIN recommendations r ON u.id = r.user_id
GROUP BY u.id, u.display_name;

CREATE VIEW IF NOT EXISTS track_popularity AS
SELECT 
    t.id,
    t.name,
    t.artist,
    t.popularity,
    COUNT(DISTINCT pt.playlist_id) as playlist_count,
    COUNT(DISTINCT r.user_id) as recommendation_count,
    AVG(r.confidence_score) as avg_confidence
FROM tracks t
LEFT JOIN playlist_tracks pt ON t.id = pt.track_id
LEFT JOIN recommendations r ON t.id = r.track_id
GROUP BY t.id, t.name, t.artist, t.popularity;
