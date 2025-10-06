#!/usr/bin/env python3
"""
Add audio feature embeddings to existing Elasticsearch tracks.
Generates 512-dim feature vectors from available metadata.
"""

import os
from elasticsearch import Elasticsearch
from cloud_run_scoring import generate_audio_embedding
from dotenv import load_dotenv

load_dotenv()

# Initialize Elasticsearch
es_url = os.getenv('ELASTICSEARCH_URL')
api_key = os.getenv('ELASTIC_API_KEY')

if not es_url or not api_key:
    print("❌ Missing Elasticsearch credentials")
    print("Set ELASTICSEARCH_URL and ELASTIC_API_KEY environment variables")
    exit(1)

es = Elasticsearch(hosts=[es_url], api_key=api_key)

# Check connection
if not es.ping():
    print("❌ Failed to connect to Elasticsearch")
    exit(1)

print(f"✅ Connected to Elasticsearch at {es_url}")

# Get all tracks
index_name = "music-tracks"

print(f"\n📊 Fetching tracks from '{index_name}'...")

# Use scroll API for large result sets
query = {"query": {"match_all": {}}}
response = es.search(index=index_name, body=query, scroll='5m', size=100)

scroll_id = response['_scroll_id']
hits = response['hits']['hits']
all_tracks = hits.copy()

print(f"Initial batch: {len(hits)} tracks")

# Continue scrolling
while len(hits) > 0:
    response = es.scroll(scroll_id=scroll_id, scroll='5m')
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    all_tracks.extend(hits)
    if hits:
        print(f"Fetched {len(hits)} more tracks (total: {len(all_tracks)})")

print(f"\n✅ Total tracks found: {len(all_tracks)}")

# Process each track
print("\n🔄 Adding audio feature embeddings...")

updated_count = 0
error_count = 0

for i, hit in enumerate(all_tracks, 1):
    track_id = hit['_id']
    track_data = hit['_source']
    
    # Check if embedding already exists
    if 'openl3' in track_data and track_data['openl3']:
        print(f"  [{i}/{len(all_tracks)}] {track_id}: Already has embedding, skipping")
        continue
    
    try:
        # Generate embedding
        embedding = generate_audio_embedding(track_data)
        
        # Update document
        es.update(
            index=index_name,
            id=track_id,
            body={
                "doc": {
                    "openl3": embedding,
                    "embedding_version": "v1.0-feature-based"
                }
            }
        )
        
        updated_count += 1
        
        # Print progress every 50 tracks
        if i % 50 == 0 or i == len(all_tracks):
            print(f"  [{i}/{len(all_tracks)}] Updated: {updated_count}, Errors: {error_count}")
        
    except Exception as e:
        error_count += 1
        print(f"  ❌ [{i}/{len(all_tracks)}] Failed to update {track_id}: {e}")

# Clear scroll
es.clear_scroll(scroll_id=scroll_id)

# Summary
print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)
print(f"Total tracks:     {len(all_tracks)}")
print(f"Updated:          {updated_count}")
print(f"Errors:           {error_count}")
print(f"Already had data: {len(all_tracks) - updated_count - error_count}")

if updated_count > 0:
    print("\n✅ Audio feature embeddings successfully added!")
    print("\nNext steps:")
    print("1. Verify with: curl -X GET \"$ELASTICSEARCH_URL/music_tracks/_search?size=1\" -H \"Authorization: ApiKey $ELASTIC_API_KEY\"")
    print("2. Test recommendations in the web UI")
    print("3. Deploy updated code to Cloud Run")
else:
    print("\nℹ️  No tracks needed updating")

print("\n" + "="*60)
