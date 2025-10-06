#!/usr/bin/env python3
"""
Verify that all infrastructure is set up correctly
"""

import os
import sys


def check_environment_variables():
    """Check if all required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    required_vars = [
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_APPLICATION_CREDENTIALS',
        'ELASTICSEARCH_URL',
        'ELASTIC_API_KEY'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
            print(f"❌ {var} not set")
        else:
            value = os.getenv(var)
            # Mask sensitive values
            if 'KEY' in var or 'CREDENTIALS' in var:
                masked = value[:10] + '...' if len(value) > 10 else '***'
                print(f"✅ {var}={masked}")
            else:
                print(f"✅ {var}={value}")
    
    if missing:
        print(f"\n❌ Missing variables: {', '.join(missing)}")
        print("Please set them in .env and run: source .env")
        return False
    
    print("✅ All environment variables are set\n")
    return True


def check_bigquery():
    """Check BigQuery connection and tables"""
    print("🔍 Checking BigQuery...")
    
    try:
        from google.cloud import bigquery
        
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        client = bigquery.Client(project=project_id)
        
        # List datasets
        datasets = list(client.list_datasets())
        print(f"✅ Connected to BigQuery project: {project_id}")
        
        # Check for music_ai dataset
        dataset_id = f"{project_id}.music_ai"
        try:
            tables = list(client.list_tables(dataset_id))
            print(f"✅ Dataset music_ai found with {len(tables)} tables:")
            
            expected_tables = ['tracks', 'audio_features', 'user_feedback', 'user_profiles']
            found_tables = [t.table_id for t in tables]
            
            for table_name in expected_tables:
                if table_name in found_tables:
                    # Get row count
                    query = f"SELECT COUNT(*) as count FROM `{dataset_id}.{table_name}`"
                    result = list(client.query(query).result())
                    count = result[0].count if result else 0
                    print(f"   ✅ {table_name} ({count} rows)")
                else:
                    print(f"   ❌ {table_name} missing")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset music_ai not found: {e}")
            print("   Run: python setup_infrastructure.py")
            return False
    
    except Exception as e:
        print(f"❌ BigQuery error: {e}")
        return False


def check_elastic():
    """Check Elastic Cloud connection and index"""
    print("\n🔍 Checking Elastic Cloud...")
    
    try:
        from elasticsearch import Elasticsearch
        
        es_url = os.getenv('ELASTICSEARCH_URL')
        api_key = os.getenv('ELASTIC_API_KEY')
        
        es = Elasticsearch(hosts=[es_url], api_key=api_key)
        
        if not es.ping():
            print("❌ Cannot connect to Elastic Cloud")
            return False
        
        print("✅ Connected to Elastic Cloud")
        
        # Check index
        index_name = "music-tracks"
        if es.indices.exists(index=index_name):
            stats = es.indices.stats(index=index_name)
            doc_count = stats['indices'][index_name]['total']['docs']['count']
            print(f"✅ Index {index_name} exists ({doc_count} documents)")
            
            # Check mapping
            mapping = es.indices.get_mapping(index=index_name)
            properties = mapping[index_name]['mappings']['properties']
            
            required_fields = ['track_id', 'title', 'artist', 'bpm', 'key', 'openl3']
            for field in required_fields:
                if field in properties:
                    field_type = properties[field].get('type', 'unknown')
                    print(f"   ✅ {field} ({field_type})")
                else:
                    print(f"   ❌ {field} missing")
            
            return True
        else:
            print(f"❌ Index {index_name} not found")
            print("   Run: python setup_infrastructure.py")
            return False
    
    except Exception as e:
        print(f"❌ Elastic error: {e}")
        return False


def check_api():
    """Check if API can start"""
    print("\n🔍 Checking API...")
    
    try:
        # Try importing main modules
        from fastapi import FastAPI
        from elasticsearch import Elasticsearch
        from google.cloud import bigquery
        print("✅ All required packages installed")
        
        # Check if main.py exists
        if os.path.exists('main.py'):
            print("✅ main.py found")
        else:
            print("❌ main.py not found")
            return False
        
        return True
    
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   Run: pip install -r requirements.txt")
        return False


def main():
    """Run all checks"""
    print("=" * 60)
    print("🎵 AI Music Discovery Engine - Setup Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check environment
    if not check_environment_variables():
        all_ok = False
    
    # Check BigQuery
    if not check_bigquery():
        all_ok = False
    
    # Check Elastic
    if not check_elastic():
        all_ok = False
    
    # Check API
    if not check_api():
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All checks passed! System is ready.")
        print("=" * 60)
        print("\n📝 Next steps:")
        print("   1. If no data: python ingest_sample_data.py")
        print("   2. Start API: python main.py")
        print("   3. Test: curl http://localhost:8080/health")
    else:
        print("❌ Some checks failed. Please fix errors above.")
        print("=" * 60)
        print("\n📝 Setup instructions:")
        print("   1. Review SETUP_GUIDE.md")
        print("   2. Run: python setup_infrastructure.py")
        print("   3. Run this script again")
        sys.exit(1)


if __name__ == "__main__":
    main()
