#!/usr/bin/env python3
"""
Check Cloud Run deployment status without gcloud CLI
"""

import requests
import sys

CLOUD_RUN_URL = "https://music-ai-backend-695632103465.us-central1.run.app"

def check_deployment():
    """Check if Cloud Run service is accessible"""
    
    print("🔍 Checking Cloud Run Deployment Status...")
    print(f"URL: {CLOUD_RUN_URL}")
    print("="*60)
    
    endpoints = [
        ("/health", "Health Check"),
        ("/", "Frontend"),
        ("/docs", "API Documentation"),
    ]
    
    for endpoint, description in endpoints:
        url = f"{CLOUD_RUN_URL}{endpoint}"
        print(f"\n📡 Testing: {description}")
        print(f"   Endpoint: {endpoint}")
        
        try:
            response = requests.get(url, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS")
                if 'json' in response.headers.get('content-type', ''):
                    print(f"   Response: {response.json()}")
                else:
                    print(f"   Response: {response.text[:200]}...")
            elif response.status_code == 404:
                print(f"   ❌ NOT FOUND - Service may not be running")
            elif response.status_code == 503:
                print(f"   ⚠️  SERVICE UNAVAILABLE - Container may be starting")
            else:
                print(f"   ⚠️  Unexpected status: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏱️  TIMEOUT - Service not responding")
        except requests.exceptions.ConnectionError as e:
            print(f"   ❌ CONNECTION ERROR: {e}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print("\n" + "="*60)
    print("📊 DIAGNOSIS:")
    print("="*60)
    
    # Try to get a response from the service
    try:
        response = requests.get(f"{CLOUD_RUN_URL}/health", timeout=5)
        if response.status_code == 404:
            print("❌ Cloud Run service is returning 404")
            print("   Possible causes:")
            print("   1. Container failed to start")
            print("   2. Application error during startup")
            print("   3. Service not deployed yet")
            print("   4. Wrong service URL")
            print("\n💡 Solutions:")
            print("   1. Check Cloud Run logs in Google Cloud Console:")
            print(f"      https://console.cloud.google.com/run/detail/us-central1/music-ai-backend/logs")
            print("   2. Rebuild and redeploy:")
            print("      gcloud builds submit --config=cloudbuild.yaml")
            print("   3. Check container locally:")
            print("      docker build -f Dockerfile.cloudrun -t test-image .")
            print("      docker run -p 8080:8080 test-image")
            return False
        elif response.status_code == 200:
            print("✅ Cloud Run service is HEALTHY and running!")
            return True
        else:
            print(f"⚠️  Cloud Run returned unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Unable to connect to Cloud Run service")
        print(f"   Error: {e}")
        return False

if __name__ == "__main__":
    success = check_deployment()
    sys.exit(0 if success else 1)
