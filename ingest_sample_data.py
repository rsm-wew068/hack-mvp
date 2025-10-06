#!/usr/bin/env python3
"""
Sample Data Ingestion Script
Ingests sample music tracks into BigQuery and Elastic Cloud

For hackathon: Uses a small curated dataset with pre-generated features
In production: Would process actual audio files with OpenL3/Essentia
"""

import os
import json
import numpy as np
from google.cloud import bigquery
from elasticsearch import Elasticsearch
from datetime import datetime
import uuid


# Sample tracks with realistic features
SAMPLE_TRACKS = [
    {
        "track_id": f"track_{str(uuid.uuid4())[:8]}",
        "title": "Midnight Dreams",
        "artist": "Lo-Fi Café",
        "license": "cc",
        "source_url": "https://freemusicarchive.org/sample1",
        "yt_video_id": "sample_video_1",
        "bpm": 82.0,
        "key": "A minor",
        "genre": "lo-fi hip hop",
        "energy": 0.3,
        "mfcc_mean": np.random.randn(13).tolist(),
        "spectral": 1200.5,
        "openl3": np.random.randn(512).tolist()
    },
    {
        "track_id": f"track_{str(uuid.uuid4())[:8]}",
        "title": "Focus Flow",
        "artist": "Study Beats",
        "license": "cc",
        "source_url": "https://freemusicarchive.org/sample2",
        "yt_video_id": "sample_video_2",
        "bpm": 85.0,
        "key": "C major",
        "genre": "lo-fi hip hop",
        "energy": 0.35,
        "mfcc_mean": np.random.randn(13).tolist(),
        "spectral": 1350.2,
        "openl3": np.random.randn(512).tolist()
    },
    {
        "track_id": f"track_{str(uuid.uuid4())[:8]}",
        "title": "Energy Rush",
        "artist": "The Uplifters",
        "license": "public_domain",
        "source_url": "https://freemusicarchive.org/sample3",
        "yt_video_id": "sample_video_3",
        "bpm": 128.0,
        "key": "D major",
        "genre": "electronic",
        "energy": 0.9,
        "mfcc_mean": np.random.randn(13).tolist(),
        "spectral": 2100.8,
        "openl3": np.random.randn(512).tolist()
    },
    {
        "track_id": f"track_{str(uuid.uuid4())[:8]}",
        "title": "Rainy Sunday",
        "artist": "Chill Collective",
        "license": "cc",
        "source_url": "https://freemusicarchive.org/sample4",
        "yt_video_id": "sample_video_4",
        "bpm": 78.0,
        "key": "A minor",
        "genre": "ambient",
        "energy": 0.25,
        "mfcc_mean": np.random.randn(13).tolist(),
        "spectral": 980.3,
        "openl3": np.random.randn(512).tolist()
    },
    {
        "track_id": f"track_{str(uuid.uuid4())[:8]}",
        "title": "Workout Warrior",
        "artist": "Pulse",
        "license": "cc",
        "source_url": "https://freemusicarchive.org/sample5",
        "yt_video_id": "sample_video_5",
        "bpm": 140.0,
        "key": "E minor",
        "genre": "edm",
        "energy": 0.95,
        "mfcc_mean": np.random.randn(13).tolist(),
        "spectral": 2500.1,
        "openl3": np.random.randn(512).tolist()
    },
    {
        "track_id": f"track_{str(uuid.uuid4())[:8]}",
        "title": "Coffee Shop Vibes",
        "artist": "Acoustic Dreams",
        "license": "cc",
        "source_url": "https://freemusicarchive.org/sample6",
        "yt_video_id": "sample_video_6",
        "bpm": 90.0,
        "key": "G major",
        "genre": "acoustic",
        "energy": 0.4,
        "mfcc_mean": np.random.randn(13).tolist(),
        "spectral": 1450.6,
        "openl3": np.random.randn(512).tolist()
    },
    {
        "track_id": f"track_{str(uuid.uuid4())[:8]}",
        "title": "Late Night Code",
        "artist": "Binary Beats",
        "license": "cc",
        "source_url": "https://freemusicarchive.org/sample7",
        "yt_video_id": "sample_video_7",
        "bpm": 80.0,
        "key": "C minor",
        "genre": "lo-fi hip hop",
        "energy": 0.28,
        "mfcc_mean": np.random.randn(13).tolist(),
        "spectral": 1100.4,
        "openl3": np.random.randn(512).tolist()
    },
    {
        "track_id": f"track_{str(uuid.uuid4())[:8]}",
        "title": "Summer Drive",
        "artist": "Indie Waves",
        "license": "cc",
        "source_url": "https://freemusicarchive.org/sample8",
        "yt_video_id": "sample_video_8",
        "bpm": 110.0,
        "key": "F major",
        "genre": "indie pop",
        "energy": 0.7,
        "mfcc_mean": np.random.randn(13).tolist(),
        "spectral": 1800.9,
        "openl3": np.random.randn(512).tolist()
    }
]


def ingest_to_bigquery(tracks):
    """Ingest sample tracks to BigQuery"""
    print("📤 Ingesting data to BigQuery...")
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    client = bigquery.Client(project=project_id)
    
    # Prepare data for tracks table
    tracks_data = []
    audio_features_data = []
    
    for track in tracks:
        tracks_data.append({
            "track_id": track["track_id"],
            "title": track["title"],
            "artist": track["artist"],
            "license": track["license"],
            "source_url": track["source_url"],
            "yt_video_id": track["yt_video_id"]
        })
        
        audio_features_data.append({
            "track_id": track["track_id"],
            "bpm": track["bpm"],
            "key": track["key"],
            "mfcc_mean": track["mfcc_mean"],
            "spectral": track["spectral"],
            "openl3": track["openl3"]
        })
    
    # Insert into tracks table
    tracks_table_id = f"{project_id}.music_ai.tracks"
    errors = client.insert_rows_json(tracks_table_id, tracks_data)
    
    if errors:
        print(f"❌ Errors inserting tracks: {errors}")
    else:
        print(f"✅ Inserted {len(tracks_data)} tracks to BigQuery")
    
    # Insert into audio_features table
    features_table_id = f"{project_id}.music_ai.audio_features"
    errors = client.insert_rows_json(features_table_id, audio_features_data)
    
    if errors:
        print(f"❌ Errors inserting audio features: {errors}")
    else:
        print(f"✅ Inserted {len(audio_features_data)} audio features to BigQuery")


def ingest_to_elastic(tracks):
    """Ingest sample tracks to Elastic Cloud"""
    print("📤 Ingesting data to Elastic Cloud...")
    
    es_url = os.getenv('ELASTICSEARCH_URL')
    api_key = os.getenv('ELASTIC_API_KEY')
    
    es = Elasticsearch(hosts=[es_url], api_key=api_key)
    
    index_name = "music-tracks"
    
    # Bulk index documents
    for track in tracks:
        doc = {
            "track_id": track["track_id"],
            "title": track["title"],
            "artist": track["artist"],
            "bpm": track["bpm"],
            "key": track["key"],
            "openl3": track["openl3"]
        }
        
        es.index(index=index_name, id=track["track_id"], document=doc)
    
    # Refresh index
    es.indices.refresh(index=index_name)
    
    # Verify
    count = es.count(index=index_name)['count']
    print(f"✅ Indexed {count} documents to Elastic Cloud")


def main():
    """Main ingestion function"""
    print("=" * 60)
    print("📊 Sample Data Ingestion")
    print("=" * 60)
    print()
    
    print(f"📝 Preparing {len(SAMPLE_TRACKS)} sample tracks...")
    print()
    
    # Ingest to BigQuery
    ingest_to_bigquery(SAMPLE_TRACKS)
    print()
    
    # Ingest to Elastic
    ingest_to_elastic(SAMPLE_TRACKS)
    print()
    
    print("=" * 60)
    print("✅ Sample data ingestion complete!")
    print("=" * 60)
    print()
    print("📝 Next steps:")
    print("   1. Verify data: python verify_data.py")
    print("   2. Start API: python main.py")
    print("   3. Build frontend: python app.py")


if __name__ == "__main__":
    main()
