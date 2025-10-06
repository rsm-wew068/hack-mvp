#!/bin/bash
# Update Cloud Run service with new image using REST API

export PATH=$PATH:$HOME/google-cloud-sdk/bin
source .env

# Get access token
TOKEN=$(gcloud auth print-access-token)

# Update service using Cloud Run API (PATCH)
curl -X PATCH \
  "https://run.googleapis.com/v2/projects/goole-hackathon/locations/us-central1/services/music-ai-backend" \
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
