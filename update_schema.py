#!/usr/bin/env python3
"""
Apply BigQuery schema updates
Adds missing columns to support FMA data and YouTube integration
"""

from google.cloud import bigquery
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'goole-hackathon')
DATASET = 'music_ai'

def update_schema():
    """Apply schema updates to BigQuery tables"""
    
    client = bigquery.Client(project=PROJECT_ID)
    
    print("🔧 Updating BigQuery schema...\n")
    
    # Update tracks table
    print("📊 Updating tracks table...")
    tracks_table_id = f'{PROJECT_ID}.{DATASET}.tracks'
    tracks_table = client.get_table(tracks_table_id)
    
    # Add new fields to tracks schema
    original_schema = tracks_table.schema
    new_schema = original_schema[:]
    
    # Fields to add (check if they don't exist)
    new_fields_tracks = [
        bigquery.SchemaField("album", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("genre", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("bpm", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("key", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("duration_seconds", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("view_count", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("like_count", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("thumbnail_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("video_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    ]
    
    existing_field_names = {field.name for field in original_schema}
    for field in new_fields_tracks:
        if field.name not in existing_field_names:
            new_schema.append(field)
            print(f"  ✅ Adding field: {field.name} ({field.field_type})")
    
    tracks_table.schema = new_schema
    tracks_table = client.update_table(tracks_table, ["schema"])
    print(f"✅ Updated tracks table with {len(new_schema)} total fields\n")
    
    # Update audio_features table
    print("🎵 Updating audio_features table...")
    features_table_id = f'{PROJECT_ID}.{DATASET}.audio_features'
    features_table = client.get_table(features_table_id)
    
    original_schema = features_table.schema
    new_schema = original_schema[:]
    
    # Fields to add
    new_fields_features = [
        bigquery.SchemaField("energy", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("danceability", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("valence", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("acousticness", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("instrumentalness", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
    ]
    
    existing_field_names = {field.name for field in original_schema}
    for field in new_fields_features:
        if field.name not in existing_field_names:
            new_schema.append(field)
            print(f"  ✅ Adding field: {field.name} ({field.field_type})")
    
    features_table.schema = new_schema
    features_table = client.update_table(features_table, ["schema"])
    print(f"✅ Updated audio_features table with {len(new_schema)} total fields\n")
    
    print("✨ Schema update complete!")
    print("\n📋 Summary:")
    print(f"  • Tracks table: {len(tracks_table.schema)} fields")
    print(f"  • Audio features table: {len(features_table.schema)} fields")
    print("\n🎯 Ready for:")
    print("  • FMA dataset ingestion (current)")
    print("  • YouTube API integration (future)")


if __name__ == '__main__':
    update_schema()
    
    print("\n✅ Next step:")
    print("  python ingest_fma_to_bigquery.py")
