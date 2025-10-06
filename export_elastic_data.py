"""
Export Elasticsearch data from local instance to JSON file.
This prepares data for migration to Elastic Cloud.
"""

from elasticsearch import Elasticsearch
import json
import os

def export_elastic_data():
    """Export all documents from local Elasticsearch"""
    
    # Connect to local Elasticsearch
    ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    
    print(f"📡 Connecting to local Elasticsearch: {ELASTICSEARCH_URL}")
    es = Elasticsearch([ELASTICSEARCH_URL])
    
    # Verify connection
    if not es.ping():
        print("❌ Cannot connect to Elasticsearch!")
        return False
    
    info = es.info()
    print(f"✅ Connected to Elasticsearch {info['version']['number']}")
    
    # Get index info
    index_name = "music-tracks"
    
    if not es.indices.exists(index=index_name):
        print(f"❌ Index '{index_name}' does not exist!")
        return False
    
    # Get document count
    count_response = es.count(index=index_name)
    total_docs = count_response['count']
    print(f"📊 Found {total_docs} documents in index")
    
    # Export mapping
    mapping = es.indices.get_mapping(index=index_name)
    with open('/tmp/music-tracks-mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Exported mapping to /tmp/music-tracks-mapping.json")
    
    # Export all documents using scroll API
    print(f"📥 Exporting documents...")
    
    docs = []
    
    # Initial search with scroll
    response = es.search(
        index=index_name,
        body={"query": {"match_all": {}}},
        scroll='2m',
        size=100  # Batch size
    )
    
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    
    # Process first batch
    for hit in hits:
        docs.append(hit['_source'])
    
    # Continue scrolling
    while len(hits) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        for hit in hits:
            docs.append(hit['_source'])
        
        print(f"  Exported {len(docs)}/{total_docs} documents...", end='\r')
    
    print(f"\n✅ Exported {len(docs)} documents")
    
    # Save to file
    output_file = '/tmp/music-tracks-data.json'
    with open(output_file, 'w') as f:
        json.dump(docs, f, indent=2)
    
    print(f"✅ Saved to {output_file}")
    
    # Show file size
    file_size = os.path.getsize(output_file)
    print(f"📦 File size: {file_size / 1024 / 1024:.2f} MB")
    
    # Clear scroll
    es.clear_scroll(scroll_id=scroll_id)
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 Elasticsearch Local Export Tool")
    print("=" * 60)
    print()
    
    success = export_elastic_data()
    
    if success:
        print()
        print("=" * 60)
        print("✅ Export completed successfully!")
        print("=" * 60)
        print()
        print("📋 Next steps:")
        print("1. Create Elastic Cloud deployment")
        print("2. Set environment variables:")
        print("   export ELASTIC_CLOUD_ID='your-cloud-id'")
        print("   export ELASTIC_CLOUD_PASSWORD='your-password'")
        print("3. Run: python import_to_elastic_cloud.py")
    else:
        print()
        print("=" * 60)
        print("❌ Export failed!")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("- Is Elasticsearch running? Check: curl localhost:9200")
        print("- Is the index created? Check: curl localhost:9200/music-tracks")
