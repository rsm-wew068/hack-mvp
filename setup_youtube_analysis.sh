#!/bin/bash
# YouTube Audio Analysis Setup and Execution Script

echo "🎵 YouTube Audio Analysis Pipeline Setup"
echo "========================================"
echo ""

# Check if ffmpeg is installed (required by yt-dlp)
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg not found. Installing..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
else
    echo "✓ ffmpeg is installed"
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -q librosa openl3 soundfile yt-dlp resampy

echo ""
echo "✓ Setup complete!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Analyze curated videos (100 tracks, ~30-60 minutes):"
echo "   python youtube_audio_analyzer.py curated_videos.json analyzed_tracks.json"
echo ""
echo "2. Ingest to Elasticsearch:"
echo "   python ingest_youtube_tracks.py analyzed_tracks.json"
echo ""
echo "3. Test the API:"
echo "   curl -X POST http://localhost:8000/api/text-to-playlist \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"query\": \"upbeat electronic dance music\", \"num_results\": 5}'"
echo ""
echo "💡 Tips:"
echo "   - Analysis takes 12-23 seconds per track"
echo "   - Progress is saved every 10 tracks"
echo "   - Failed videos are logged separately"
echo "   - Cost: ~$0.16 for 100 tracks"
echo ""
