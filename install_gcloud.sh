#!/bin/bash
# Quick gcloud CLI installation for Linux

echo "📦 Installing Google Cloud SDK..."

# Download and install
curl https://sdk.cloud.google.com | bash

# Restart shell or source
exec -l $SHELL

echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Restart your terminal or run: source ~/.bashrc"
echo "   2. Initialize: gcloud init"
echo "   3. Login: gcloud auth login"
