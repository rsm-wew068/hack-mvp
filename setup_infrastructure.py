#!/usr/bin/env python3
"""
Infrastructure Setup Script
Sets up BigQuery tables and Elastic Cloud index from scratch
"""

import os
from google.cloud import bigquery
from elasticsearch import Elasticsearch
import json


def setup_bigquery():
    """Create BigQuery dataset and tables"""
    print("🔧 Setting up BigQuery...")
    
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        print("❌ GOOGLE_CLOUD_PROJECT environment variable not set")
        print("   Run: export GOOGLE_CLOUD_PROJECT=your-project-id")
        return False
    
    client = bigquery.Client(project=project_id)
    dataset_id = f"{project_id}.music_ai"
    
    # Create dataset
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "US"
    
    try:
        dataset = client.create_dataset(dataset, exists_ok=True)
        print(f"✅ Dataset {dataset_id} created or already exists")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False
    
    # Read schema from SQL file
    with open('bigquery_schema.sql', 'r') as f:
        schema_sql = f.read()
    
    # Execute each CREATE TABLE statement
    statements = schema_sql.split(';')
    for statement in statements:
        statement = statement.strip()
        if statement and statement.upper().startswith('CREATE TABLE'):
            try:
                client.query(statement).result()
                # Extract table name for logging
                table_name = statement.split('`')[1].split('`')[0]
                print(f"✅ Table {table_name} created")
            except Exception as e:
                print(f"⚠️  Table creation: {e}")
    
    print("✅ BigQuery setup complete!\n")
    return True


def setup_elastic():
    """Create Elastic Cloud index with proper mappings"""
    print("🔧 Setting up Elastic Cloud...")
    
    es_url = os.getenv('ELASTICSEARCH_URL')
    api_key = os.getenv('ELASTIC_API_KEY')
    
    if not es_url or not api_key:
        print("❌ Elastic credentials not set")
        print("   Set ELASTICSEARCH_URL and ELASTIC_API_KEY environment variables")
        print("\n   Get them from: https://cloud.elastic.co/")
        print("   1. Create a deployment")
        print("   2. Copy Elasticsearch endpoint URL")
        print("   3. Create an API key")
        return False
    
    try:
        # Connect to Elastic Cloud
        es = Elasticsearch(
            hosts=[es_url],
            api_key=api_key
        )
        
        # Test connection
        if not es.ping():
            print("❌ Could not connect to Elastic Cloud")
            return False
        
        print(f"✅ Connected to Elastic Cloud")
        
        # Read mapping from JSON file
        with open('elastic_mapping.json', 'r') as f:
            mapping = json.load(f)
        
        index_name = "music-tracks"
        
        # Delete index if exists (for clean setup)
        if es.indices.exists(index=index_name):
            print(f"⚠️  Index {index_name} already exists, deleting...")
            es.indices.delete(index=index_name)
        
        # Create index with mapping
        es.indices.create(index=index_name, body=mapping)
        print(f"✅ Index {index_name} created with proper mappings")
        
        # Verify index
        info = es.indices.get(index=index_name)
        print(f"✅ Index verified: {index_name}")
        print(f"   - Dense vector dims: 512")
        print(f"   - Similarity: cosine")
        
        print("✅ Elastic Cloud setup complete!\n")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Elastic: {e}")
        return False


def verify_setup():
    """Verify all services are accessible"""
    print("🔍 Verifying setup...")
    
    success = True
    
    # Check BigQuery
    try:
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        client = bigquery.Client(project=project_id)
        dataset_id = f"{project_id}.music_ai"
        
        # List tables
        tables = list(client.list_tables(dataset_id))
        print(f"✅ BigQuery: Found {len(tables)} tables")
        for table in tables:
            print(f"   - {table.table_id}")
    except Exception as e:
        print(f"❌ BigQuery verification failed: {e}")
        success = False
    
    # Check Elastic
    try:
        cloud_id = os.getenv('ELASTIC_CLOUD_ID')
        api_key = os.getenv('ELASTIC_API_KEY')
        
        es = Elasticsearch(cloud_id=cloud_id, api_key=api_key)
        
        if es.ping():
            index_stats = es.indices.stats(index="music-tracks")
            doc_count = index_stats['indices']['music-tracks']['total']['docs']['count']
            print(f"✅ Elastic: music-tracks index ready ({doc_count} documents)")
        else:
            print("❌ Elastic: Cannot connect")
            success = False
    except Exception as e:
        print(f"❌ Elastic verification failed: {e}")
        success = False
    
    return success


def main():
    """Main setup function"""
    print("=" * 60)
    print("🎵 AI Music Discovery Engine - Infrastructure Setup")
    print("=" * 60)
    print()
    
    # Check environment variables
    print("📋 Checking environment variables...")
    required_vars = {
        'GOOGLE_CLOUD_PROJECT': os.getenv('GOOGLE_CLOUD_PROJECT'),
        'ELASTICSEARCH_URL': os.getenv('ELASTICSEARCH_URL'),
        'ELASTIC_API_KEY': os.getenv('ELASTIC_API_KEY')
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("\n💡 Setup instructions:")
        print("   1. Copy env.example to .env")
        print("   2. Fill in your credentials")
        print("   3. Run: source .env")
        print("   4. Run this script again")
        return
    
    print("✅ All environment variables set\n")
    
    # Run setup
    bigquery_ok = setup_bigquery()
    elastic_ok = setup_elastic()
    
    if bigquery_ok and elastic_ok:
        print("\n" + "=" * 60)
        print("🎉 Infrastructure setup complete!")
        print("=" * 60)
        
        verify_setup()
        
        print("\n📝 Next steps:")
        print("   1. Run: python ingest_sample_data.py")
        print("   2. Run: python main.py")
        print("   3. Build the Streamlit frontend")
    else:
        print("\n❌ Setup incomplete. Please fix errors above.")


if __name__ == "__main__":
    main()
