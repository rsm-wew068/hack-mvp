#!/usr/bin/env python3
"""
Ingest analyzed YouTube tracks to Elasticsearch.
"""

import os
import json
import logging
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_youtube_tracks(tracks_file: str):
    """
    Ingest analyzed YouTube tracks to Elasticsearch.
    
    Args:
        tracks_file: Path to JSON file with analyzed tracks
    """
    # Connect to Elasticsearch
    es_cloud_id = os.getenv("ELASTIC_CLOUD_ID")
    es_api_key = os.getenv("ELASTIC_API_KEY")
    
    if not es_cloud_id or not es_api_key:
        logger.error("Missing Elasticsearch credentials")
        return
    
    es = Elasticsearch(
        cloud_id=es_cloud_id,
        api_key=es_api_key
    )
    
    # Load tracks
    with open(tracks_file, 'r') as f:
        tracks = json.load(f)
    
    logger.info(f"Loaded {len(tracks)} tracks from {tracks_file}")
    
    index_name = "music_tracks"
    
    # Check if index exists
    if es.indices.exists(index=index_name):
        logger.info(f"Index {index_name} exists, deleting old data...")
        es.indices.delete(index=index_name)
    
    # Create index with mapping
    mapping = {
        "mappings": {
            "properties": {
                "video_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "artist": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "bpm": {"type": "float"},
                "key": {"type": "keyword"},
                "energy": {"type": "float"},
                "genre": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 512,
                    "index": True,
                    "similarity": "cosine"
                },
                "youtube_data": {
                    "properties": {
                        "view_count": {"type": "long"},
                        "like_count": {"type": "long"},
                        "duration_seconds": {"type": "integer"},
                        "thumbnail_url": {"type": "keyword"},
                        "watch_url": {"type": "keyword"}
                    }
                },
                "analysis_source": {"type": "keyword"}
            }
        }
    }
    
    es.indices.create(index=index_name, body=mapping)
    logger.info(f"Created index {index_name}")
    
    # Prepare bulk documents
    actions = []
    for track in tracks:
        # Extract genre from curated list (if available)
        # Otherwise use a default
        genre = track.get("genre", "Unknown")
        
        doc = {
            "_index": index_name,
            "_source": {
                "video_id": track["video_id"],
                "title": track["title"],
                "artist": track["artist"],
                "bpm": track["bpm"],
                "key": track["key"],
                "energy": track.get("energy", 0.5),
                "genre": genre,
                "embedding": track["embedding"],
                "youtube_data": track.get("youtube_data", {}),
                "analysis_source": "real_audio"
            }
        }
        actions.append(doc)
    
    # Bulk index
    success, failed = helpers.bulk(
        es,
        actions,
        raise_on_error=False,
        raise_on_exception=False
    )
    
    logger.info(f"✓ Successfully indexed {success} documents")
    if failed:
        logger.warning(f"Failed to index {len(failed)} documents")
    
    # Verify
    es.indices.refresh(index=index_name)
    count = es.count(index=index_name)["count"]
    logger.info(f"Total documents in index: {count}")
    
    # Show sample
    logger.info("\nSample tracks:")
    results = es.search(
        index=index_name,
        body={
            "query": {"match_all": {}},
            "size": 3
        }
    )
    
    for hit in results["hits"]["hits"]:
        track = hit["_source"]
        logger.info(
            f"  • {track['artist']} - {track['title']} "
            f"(BPM: {track['bpm']}, Key: {track['key']}, "
            f"Energy: {track['energy']:.2f})"
        )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ingest_youtube_tracks.py <analyzed_tracks.json>")
        print("Example: python ingest_youtube_tracks.py analyzed_tracks.json")
        sys.exit(1)
    
    ingest_youtube_tracks(sys.argv[1])
