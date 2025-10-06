#!/bin/bash

# Force new Cloud Run deployment
PROJECT_ID="goole-hackathon"
SERVICE_NAME="music-ai-backend"
REGION="us-central1"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "Forcing new deployment with latest image..."

curl -X PATCH \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/services/${SERVICE_NAME}" \
  -d "{
    \"template\": {
      \"containers\": [{
        \"image\": \"${IMAGE}@sha256:d09f9456468db9b37f360e2da281a46c4780d3e903988726defda6302cd29a72\"
      }],
      \"annotations\": {
        \"run.googleapis.com/client-name\": \"manual-update-$(date +%s)\"
      }
    }
  }"
