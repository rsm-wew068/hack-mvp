#!/usr/bin/env python3
"""
Ingest FMA dataset to Elasticsearch
"""

import json
from elasticsearch import Elasticsearch
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration
INPUT_FILE = './fma_processed_tracks.json'
ELASTIC_ENDPOINT = os.getenv('ELASTICSEARCH_URL')  # Use correct env var
ELASTIC_API_KEY = os.getenv('ELASTIC_API_KEY')
INDEX_NAME = 'music-tracks'


def load_tracks():
    """Load processed FMA tracks"""
    print(f"📂 Loading tracks from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        tracks = json.load(f)
    print(f"✅ Loaded {len(tracks)} tracks")
    return tracks


def ingest_to_elasticsearch(tracks):
    """Ingest tracks to Elasticsearch"""
    
    print(f"\n🚀 Ingesting {len(tracks)} tracks to Elasticsearch...")
    
    # Connect to Elasticsearch
    es = Elasticsearch(
        ELASTIC_ENDPOINT,
        api_key=ELASTIC_API_KEY
    )
    
    print(f"✅ Connected to Elasticsearch")
    
    # Delete existing documents (clean start)
    print(f"\n🗑️  Deleting existing documents in {INDEX_NAME}...")
    try:
        es.delete_by_query(
            index=INDEX_NAME,
            body={"query": {"match_all": {}}}
        )
        print(f"✅ Deleted old documents")
    except Exception as e:
        print(f"⚠️  No existing documents or error: {e}")
    
    # Prepare bulk operations
    bulk_data = []
    for track in tracks:
        # Index operation
        bulk_data.append({
            'index': {
                '_index': INDEX_NAME,
                '_id': track['track_id']
            }
        })
        
        # Document data
        doc = {
            'track_id': track['track_id'],
            'title': track['title'],
            'artist': track['artist'],
            'album': track.get('album', 'Unknown'),
            'genre': track.get('genre', 'Unknown'),
            'bpm': float(track['bpm']),
            'key': track['key'],
            'openl3': track['openl3'],
            'energy': float(track.get('energy', 0.5)),
            'danceability': float(track.get('danceability', 0.5)),
            'valence': float(track.get('valence', 0.5)),
            'yt_video_id': None,
            'created_at': track['created_at']
        }
        bulk_data.append(doc)
    
    # Bulk insert
    print(f"\n📤 Bulk indexing {len(tracks)} documents...")
    
    try:
        from elasticsearch.helpers import bulk
        
        actions = []
        for track in tracks:
            action = {
                '_index': INDEX_NAME,
                '_id': track['track_id'],
                '_source': {
                    'track_id': track['track_id'],
                    'title': track['title'],
                    'artist': track['artist'],
                    'album': track.get('album', 'Unknown'),
                    'genre': track.get('genre', 'Unknown'),
                    'bpm': float(track['bpm']),
                    'key': track['key'],
                    'openl3': track['openl3'],
                    'energy': float(track.get('energy', 0.5)),
                    'danceability': float(track.get('danceability', 0.5)),
                    'valence': float(track.get('valence', 0.5)),
                    'yt_video_id': None,
                    'created_at': track['created_at']
                }
            }
            actions.append(action)
        
        success, failed = bulk(es, actions)
        print(f"✅ Successfully indexed {success} documents")
        if failed:
            print(f"❌ Failed to index {len(failed)} documents")
    
    except Exception as e:
        print(f"❌ Bulk indexing error: {e}")
    
    # Refresh index
    es.indices.refresh(index=INDEX_NAME)
    
    # Verify count
    print(f"\n🔍 Verifying data...")
    count = es.count(index=INDEX_NAME)['count']
    print(f"  Elasticsearch index: {count} documents")
    
    print(f"\n✅ Elasticsearch ingestion complete!")


if __name__ == '__main__':
    tracks = load_tracks()
    ingest_to_elasticsearch(tracks)
    
    print(f"\n✨ DONE! {len(tracks)} tracks now in Elasticsearch")
    print(f"\nNext step:")
    print(f"  ./start_services.sh")
    print(f"\nOr test the API:")
    print(f"  curl http://localhost:8080/health")
