# 🎵 YouTube Audio Analysis Implementation

## Overview

This implementation uses **real audio analysis** from YouTube videos to extract music features:
- **BPM (Tempo)**: Detected using Librosa beat tracking
- **Musical Key**: Detected using chroma features
- **Energy Level**: Calculated from RMS amplitude
- **512-dim Embeddings**: Generated with OpenL3 music model

## Architecture

```
YouTube Video → yt-dlp → WAV Audio → Librosa + OpenL3 → Features → Elasticsearch
                                            ↓
                                    BPM, Key, Energy
                                    512-dim embedding
```

## Cost Analysis (100 Tracks)

| Component | Cost | Notes |
|-----------|------|-------|
| YouTube API | $0.00 | Search is free, watch is free |
| Audio Download | $0.00 | yt-dlp is free |
| Cloud Run Analysis | $0.16 | 100 tracks × 20s × $0.00002400/vCPU-sec |
| Elasticsearch | $0.00 | Free tier sufficient |
| **Total Initial** | **$0.16** | One-time analysis cost |
| **Monthly** | **$5-10** | ~30 new tracks/month on-demand |

## Setup

### 1. Install Dependencies

```bash
bash setup_youtube_analysis.sh
```

Or manually:
```bash
# Install ffmpeg (required by yt-dlp)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Install Python packages
pip install librosa openl3 soundfile yt-dlp resampy
```

### 2. Analyze Curated Videos (100 tracks)

```bash
python youtube_audio_analyzer.py curated_videos.json analyzed_tracks.json
```

**Expected time**: 30-60 minutes (12-23 seconds per track)

**What it does**:
- Downloads audio from YouTube (30 seconds)
- Analyzes BPM using beat tracking
- Detects musical key using chroma features
- Calculates energy/loudness
- Generates OpenL3 512-dim embedding
- Saves progress every 10 tracks

### 3. Ingest to Elasticsearch

```bash
python ingest_youtube_tracks.py analyzed_tracks.json
```

**What it does**:
- Creates/updates `music_tracks` index
- Uploads all analyzed tracks
- Configures dense vector search for embeddings
- Enables hybrid BM25 + semantic search

### 4. Test the API

```bash
# Start local server
uvicorn main:app --reload

# Test search
curl -X POST http://localhost:8000/api/text-to-playlist \
  -H 'Content-Type: application/json' \
  -d '{"query": "upbeat electronic dance music", "num_results": 5}'
```

## On-Demand Analysis

For tracks not in the catalog, you can analyze on-demand:

```python
# In main.py, add this endpoint:
@app.post("/api/analyze-youtube")
async def analyze_youtube(request: AnalyzeRequest):
    """Analyze a YouTube video in real-time"""
    # See on_demand_analysis.py for implementation
    pass
```

**Use case**: User searches for a song not in catalog
1. Search YouTube for the song
2. Analyze audio in real-time (15-30 seconds)
3. Add to Elasticsearch
4. Return in search results
5. Cache forever (no re-analysis needed)

## Data Quality

### Real Audio Analysis vs Spotify API

| Feature | YouTube Analysis | Spotify API |
|---------|------------------|-------------|
| **BPM** | ✅ Real detection | ✅ Perfect |
| **Key** | ✅ Real detection | ✅ Perfect |
| **Energy** | ✅ Calculated | ✅ Perfect |
| **Embeddings** | ✅ OpenL3 (512-dim) | ❌ Not available |
| **Cost** | $0.0016/track | $0.00 (free) |
| **Speed** | 12-23 seconds | <500ms |
| **Catalog** | Any YouTube video | 100M+ tracks |

### Why YouTube Analysis is Better for This Hackathon

1. **🎯 Real Audio Processing**: Shows actual ML/AI capability
2. **🚀 Custom Features**: OpenL3 embeddings for similarity
3. **📖 Great Story**: "We analyze the actual audio in real-time"
4. **💰 Affordable**: $0.16 for 100 tracks, $5-10/month ongoing
5. **🎬 Demo-Worthy**: Can show "analyzing audio..." live

## Curated Video List

The `curated_videos.json` contains 100 diverse tracks:
- 15+ genres (Pop, Rock, Electronic, Hip-Hop, Indie, etc.)
- Tempo variety (60-180 BPM estimated)
- Popular + diverse artists
- All have official YouTube videos

**Genres covered**:
- Pop/Dance: Shakira, Lady Gaga, Taylor Swift
- Electronic/EDM: Avicii, Daft Punk, The Chainsmokers
- Rock: Queen, Nirvana, Linkin Park
- Hip-Hop: Eminem, Drake
- Indie: Gotye, Hozier, Passenger
- R&B/Soul: Adele, Bruno Mars, Rihanna
- And more...

## Troubleshooting

### Download Failures
Some videos may fail due to:
- Geographic restrictions
- Age restrictions
- Copyright blocks

**Solution**: Script automatically logs failed videos to `*_failed.json`

### Analysis Errors
If BPM/key detection fails:
- Audio too short
- Non-music content
- Very complex music

**Solution**: Falls back to defaults (logged as warning)

### Memory Issues
OpenL3 requires ~2GB RAM per track

**Solution**: Process in batches, or use Cloud Run with 4GB memory

## Scaling Beyond 100 Tracks

| Tracks | Analysis Time | Storage | Monthly Cost |
|--------|---------------|---------|--------------|
| 100 | 30-60 min | 50 MB | $5-10 |
| 500 | 2.5-5 hours | 250 MB | $15-25 |
| 1,000 | 5-10 hours | 500 MB | $50-60 |
| 5,000 | 1-2 days | 2.5 GB | $200-250 |

**Recommendation**: Start with 100, grow based on usage patterns

## License & Legal

⚠️ **Important**: Downloading copyrighted content from YouTube may violate Terms of Service.

**For hackathon/research**: Generally acceptable
**For production**: Consider:
- Spotify API (100% licensed, free)
- Licensed audio providers
- User-uploaded content only
- YouTube API + metadata only (no download)

## Next Steps

1. ✅ Run `bash setup_youtube_analysis.sh`
2. ✅ Analyze 100 curated videos (~45 minutes)
3. ✅ Ingest to Elasticsearch
4. ✅ Deploy to Cloud Run
5. ✅ Demo real-time audio analysis!

---

**Total Cost for Hackathon**: ~$6 ($0.16 initial + $5 month 1)

**Demo Impact**: 🚀🚀🚀 Show real AI audio analysis!
