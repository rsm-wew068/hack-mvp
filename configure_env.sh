#!/bin/bash
# Interactive setup script for environment configuration

echo "=================================================="
echo "🎵 AI Music Discovery Engine - Environment Setup"
echo "=================================================="
echo ""

# Create .env file
ENV_FILE=".env"

echo "Let's configure your environment variables."
echo ""

# Google Cloud Project
read -p "Enter your Google Cloud Project ID: " GCP_PROJECT
echo "export GOOGLE_CLOUD_PROJECT=$GCP_PROJECT" > $ENV_FILE

# Credentials path
CREDS_PATH="$(pwd)/credentials.json"
echo "export GOOGLE_APPLICATION_CREDENTIALS=$CREDS_PATH" >> $ENV_FILE

# BigQuery
echo "export BIGQUERY_DATASET=music_ai" >> $ENV_FILE
echo "export BIGQUERY_LOCATION=US" >> $ENV_FILE

# Elastic Cloud
echo ""
echo "Now, let's configure Elastic Cloud:"
echo "1. Go to your Elastic deployment"
echo "2. Click 'Manage' to get your Cloud ID"
echo "3. Go to Security → API Keys to create/get an API key"
echo ""

read -p "Enter your Elastic Cloud ID: " ELASTIC_CLOUD_ID
read -p "Enter your Elastic API Key: " ELASTIC_API_KEY

echo "export ELASTIC_CLOUD_ID='$ELASTIC_CLOUD_ID'" >> $ENV_FILE
echo "export ELASTIC_API_KEY='$ELASTIC_API_KEY'" >> $ENV_FILE

# Audio settings
echo "export AUDIO_SAMPLE_RATE=22050" >> $ENV_FILE
echo "export AUDIO_DURATION=30" >> $ENV_FILE
echo "export OPENL3_EMBEDDING_SIZE=512" >> $ENV_FILE

# App settings
echo "export DEBUG=False" >> $ENV_FILE
echo "export LOG_LEVEL=INFO" >> $ENV_FILE

echo ""
echo "✅ Environment file created: .env"
echo ""
echo "📝 Next steps:"
echo "   1. Make sure you have credentials.json in this directory"
echo "   2. Run: source .env"
echo "   3. Run: python verify_setup.py"
echo ""
echo "Your .env file:"
echo "=================================================="
cat $ENV_FILE
echo "=================================================="
