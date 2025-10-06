#!/usr/bin/env python3
"""
Ingest FMA dataset to BigQuery
"""

import json
from google.cloud import bigquery
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
INPUT_FILE = './fma_processed_tracks.json'
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'goole-hackathon')
DATASET = 'music_ai'
TRACKS_TABLE = f'{PROJECT_ID}.{DATASET}.tracks'
FEATURES_TABLE = f'{PROJECT_ID}.{DATASET}.audio_features'


def load_tracks():
    """Load processed FMA tracks"""
    print(f"📂 Loading tracks from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        tracks = json.load(f)
    print(f"✅ Loaded {len(tracks)} tracks")
    return tracks


def ingest_to_bigquery(tracks):
    """Ingest tracks to BigQuery"""
    
    print(f"\n🚀 Ingesting {len(tracks)} tracks to BigQuery...")
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # Prepare tracks data
    tracks_data = []
    features_data = []
    
    for track in tracks:
        # Tracks table
        track_row = {
            'track_id': track['track_id'],
            'title': track['title'],
            'artist': track['artist'],
            'license': track.get('license', 'Unknown'),  # FMA license info
            'source_url': track.get('source_url', f'https://freemusicarchive.org/music/{track.get("artist", "unknown")}'),
            'album': track.get('album', 'Unknown'),
            'genre': track.get('genre', 'Unknown'),
            'bpm': float(track['bpm']),
            'key': track['key'],
            'yt_video_id': None,  # Will be filled later with YouTube API
            'created_at': track['created_at']
        }
        tracks_data.append(track_row)
        
        # Audio features table
        features_row = {
            'track_id': track['track_id'],
            'openl3': track['openl3'],
            'energy': float(track.get('energy', 0.5)),
            'danceability': float(track.get('danceability', 0.5)),
            'valence': float(track.get('valence', 0.5)),
            'created_at': track['created_at']
        }
        features_data.append(features_row)
    
    # Insert into tracks table
    print(f"\n📊 Inserting {len(tracks_data)} rows into tracks table...")
    errors = client.insert_rows_json(TRACKS_TABLE, tracks_data)
    if errors:
        print(f"❌ Errors inserting tracks: {errors}")
    else:
        print(f"✅ Successfully inserted {len(tracks_data)} tracks")
    
    # Insert into audio_features table
    print(f"\n🎵 Inserting {len(features_data)} rows into audio_features table...")
    errors = client.insert_rows_json(FEATURES_TABLE, features_data)
    if errors:
        print(f"❌ Errors inserting features: {errors}")
    else:
        print(f"✅ Successfully inserted {len(features_data)} audio features")
    
    # Verify counts
    print(f"\n🔍 Verifying data...")
    
    tracks_count = client.query(
        f"SELECT COUNT(*) as count FROM `{TRACKS_TABLE}`"
    ).result()
    for row in tracks_count:
        print(f"  Tracks table: {row['count']} rows")
    
    features_count = client.query(
        f"SELECT COUNT(*) as count FROM `{FEATURES_TABLE}`"
    ).result()
    for row in features_count:
        print(f"  Features table: {row['count']} rows")
    
    print(f"\n✅ BigQuery ingestion complete!")


if __name__ == '__main__':
    tracks = load_tracks()
    ingest_to_bigquery(tracks)
    
    print(f"\n✨ DONE! {len(tracks)} tracks now in BigQuery")
    print(f"\nNext step:")
    print(f"  python ingest_fma_to_elasticsearch.py")
