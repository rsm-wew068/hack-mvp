#!/bin/bash
# Deploy to Cloud Run using REST API (bypasses gcloud grpc issue)

export PATH=$PATH:$HOME/google-cloud-sdk/bin
source .env

# Get access token
TOKEN=$(gcloud auth print-access-token)

# Deploy using Cloud Run API
curl -X POST \
  "https://run.googleapis.com/v2/projects/goole-hackathon/locations/us-central1/services?serviceId=music-ai-backend" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template": {
      "containers": [{
        "image": "gcr.io/goole-hackathon/music-ai-backend:latest",
        "ports": [{"containerPort": 8080}],
        "env": [
          {"name": "GOOGLE_CLOUD_PROJECT", "value": "goole-hackathon"},
          {"name": "ELASTICSEARCH_URL", "value": "'"$ELASTICSEARCH_URL"'"},
          {"name": "ELASTIC_API_KEY", "value": "'"$ELASTIC_API_KEY"'"}
        ],
        "resources": {
          "limits": {
            "memory": "2Gi",
            "cpu": "2"
          }
        }
      }],
      "timeout": "300s"
    }
  }' | python -m json.tool
