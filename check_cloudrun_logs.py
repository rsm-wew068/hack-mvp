#!/usr/bin/env python3
"""
Check Cloud Run service status and recent logs
Uses REST API (no gcloud CLI needed)
"""
import requests
import json
from datetime import datetime

SERVICE_URL = "https://music-ai-backend-695632103465.us-central1.run.app"

def check_service():
    """Check if service is responding"""
    print(f"\n{'='*60}")
    print(f"Checking Cloud Run Service: {SERVICE_URL}")
    print(f"{'='*60}\n")
    
    endpoints = [
        '/health',
        '/',
        '/api/search'
    ]
    
    for endpoint in endpoints:
        url = f"{SERVICE_URL}{endpoint}"
        try:
            print(f"Testing: {endpoint}")
            response = requests.get(url, timeout=10)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"  ✅ SUCCESS!")
                if response.headers.get('content-type', '').startswith('application/json'):
                    data = response.json()
                    print(f"  Response: {json.dumps(data, indent=2)[:200]}")
            elif response.status_code == 404:
                print(f"  ❌ NOT FOUND (container may not be running)")
            else:
                print(f"  ⚠️  Error: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print(f"  ⏱️  TIMEOUT (container may be starting or crashed)")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        
        print()

if __name__ == "__main__":
    check_service()
    print("\n💡 To see detailed logs, use:")
    print("   gcloud logging read \"resource.type=cloud_run_revision\" --limit 50")
