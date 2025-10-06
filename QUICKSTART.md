# 🚀 Quick Start: YouTube Audio Analysis# 🚀 Quick Start Guide



Get your music AI running with **real audio analysis** in under 1 hour!This guide will get you from zero to a working AI Music Discovery Engine in **30-40 minutes**.



## TL;DR## 📋 Prerequisites Checklist

```bash

# 1. Setup (2 minutes)- [ ] Python 3.11+ installed

bash setup_youtube_analysis.sh- [ ] Google Cloud account (with billing enabled)

- [ ] Terminal access

# 2. Analyze 100 tracks (45 minutes)

python youtube_audio_analyzer.py curated_videos.json analyzed_tracks.json## ⚡ Fast Track Setup



# 3. Ingest to Elasticsearch (1 minute)### 1. Install Dependencies (2 min)

python ingest_youtube_tracks.py analyzed_tracks.json```bash

cd /home/jovyan/git/hack-mvp

# 4. Deploy (5 minutes)pip install -r requirements.txt

bash update_cloudrun.sh```



# Done! 🎉### 2. Set Up Google Cloud (10 min)

``````bash

# Login to Google Cloud

## Step-by-Stepgcloud auth login

gcloud config set project YOUR_PROJECT_ID

### Prerequisites

- ✅ Google Cloud project with billing# Enable APIs

- ✅ Elasticsearch cloud credentials in `.env`gcloud services enable bigquery.googleapis.com storage.googleapis.com aiplatform.googleapis.com run.googleapis.com

- ✅ Python 3.11+

- ✅ ~2GB disk space# Set up auth

gcloud auth application-default login

### Step 1: Install Dependencies (2 min)```

```bash

bash setup_youtube_analysis.sh### 3. Set Up Elastic Cloud (10 min)

```1. Go to https://cloud.elastic.co/registration

2. Create free account and deployment

**What this does**:3. Copy **Cloud ID** and create **API Key**

- Installs ffmpeg (audio codec)

- Installs librosa (BPM/key detection)### 4. Configure Environment (2 min)

- Installs openl3 (embeddings)```bash

- Installs yt-dlp (YouTube downloader)# Copy template

cp .env.template .env

### Step 2: Analyze Curated Videos (45 min)

```bash# Edit with your credentials

python youtube_audio_analyzer.py curated_videos.json analyzed_tracks.jsonnano .env  # or your preferred editor

```

# Load environment

**What happens**:source .env

``````

[1/100] Processing Rick Astley - Never Gonna Give You Up

  Downloading audio...### 5. Run Setup Script (5 min)

  Analyzing BPM... 113.3```bash

  Detecting key... A major# Creates BigQuery tables and Elastic index

  Calculating energy... 0.782python setup_infrastructure.py

  Generating OpenL3 embedding... (512,)```

✓ Successfully analyzed Rick Astley - Never Gonna Give You Up

### 6. Load Sample Data (2 min)

[2/100] Processing PSY - Gangnam Style```bash

...# Loads 8 sample music tracks

```python ingest_sample_data.py

```

**Progress saved every 10 tracks** - you can stop and resume!

### 7. Verify Everything Works (1 min)

**Typical timing**:```bash

- Fast tracks (simple): 12-15 seconds# Run verification

- Medium tracks: 18-20 secondspython verify_setup.py

- Slow tracks (complex): 23-30 seconds```

- **Average**: ~27 seconds per track = 45 minutes total

You should see all ✅ checks pass!

### Step 3: Ingest to Elasticsearch (1 min)

```bash### 8. Start the API (1 min)

python ingest_youtube_tracks.py analyzed_tracks.json```bash

```# Start FastAPI server

python main.py

**Output**:```

```

Loaded 100 tracks from analyzed_tracks.jsonAPI will be running at http://localhost:8080

Created index music_tracks

✓ Successfully indexed 100 documents### 9. Test It! (1 min)

Total documents in index: 100```bash

# In another terminal

Sample tracks:curl http://localhost:8080/health

  • Rick Astley - Never Gonna Give You Up (BPM: 113.3, Key: A major, Energy: 0.78)

  • PSY - Gangnam Style (BPM: 132.0, Key: D minor, Energy: 0.91)# Test recommendations

  • Luis Fonsi - Despacito (BPM: 89.0, Key: B minor, Energy: 0.68)curl -X POST http://localhost:8080/api/recommend \

```  -H "Content-Type: application/json" \

  -d '{"query": "lo-fi beats for studying", "top_k": 5}'

### Step 4: Test Locally (optional)```

```bash

# Start server## 🎉 Success!

uvicorn main:app --reload

If you see JSON responses with music recommendations, you're ready to build the frontend!

# In another terminal, test

curl -X POST http://localhost:8000/api/text-to-playlist \## 📝 Next: Build Frontend

  -H 'Content-Type: application/json' \

  -d '{"query": "energetic dance music", "num_results": 5}'Now choose your frontend:

```

### Option A: Streamlit (Recommended - 4-6 hours)

### Step 5: Deploy to Cloud Run (5 min)```bash

```bash# Create Streamlit app

bash update_cloudrun.shstreamlit run app.py

``````



**What this does**:### Option B: Gradio (Fastest - 2-3 hours)

- Builds Docker image with audio libraries```bash

- Pushes to Google Container Registry# Launch Gradio interface

- Deploys to Cloud Run (4GB memory for audio processing)python gradio_app.py

- Configures auto-scaling```



### Step 6: Test Production### Option C: Next.js (Best looking - 8-12 hours)

```bash```bash

CLOUD_RUN_URL="https://music-ai-backend-695632103465.us-central1.run.app"cd frontend

npm install

curl -X POST $CLOUD_RUN_URL/api/text-to-playlist \npm run dev

  -H 'Content-Type: application/json' \```

  -d '{

    "query": "upbeat electronic for coding",## 🐛 Troubleshooting

    "num_results": 5

  }'### "Cannot connect to BigQuery"

``````bash

gcloud auth application-default login

**Expected response**:export GOOGLE_CLOUD_PROJECT=your-project-id

```json```

{

  "recommendations": [### "Cannot connect to Elastic"

    {```bash

      "track": {# Verify credentials

        "title": "Levels",echo $ELASTIC_CLOUD_ID

        "artist": "Avicii",echo $ELASTIC_API_KEY

        "bpm": 126.0,

        "key": "C minor",# Re-source environment

        "energy": 0.89,source .env

        "genre": "Electronic/House"```

      },

      "score": 0.94,### "No module named 'X'"

      "explanation": "Perfect for coding! High energy (0.89) electronic house track at 126 BPM creates focused, uplifting atmosphere."```bash

    }pip install -r requirements.txt

    ...```

  ]

}## 📚 Full Documentation

```

- [Complete Setup Guide](SETUP_GUIDE.md) - Detailed instructions

## What You Get- [Architecture](README.md#architecture) - System design

- [API Documentation](http://localhost:8080/docs) - After starting API

✅ **100 real analyzed tracks** with:

- Real BPM from beat tracking## 🆘 Need Help?

- Musical key from chroma analysis

- Energy levels from amplitudeRun the verification script to diagnose issues:

- 512-dimensional OpenL3 embeddings```bash

python verify_setup.py

✅ **Hybrid search** powered by:```

- Elasticsearch BM25 (keyword matching)

- Dense vector search (semantic similarity)It will tell you exactly what's missing or misconfigured.

- Google Cloud AI explanations

---

✅ **On-demand analysis** capability:

- Analyze new tracks in real-time**Ready to build the frontend?** Let's do it! 🚀

- Add to catalog automatically
- Cache forever (no re-analysis)

## Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Initial analysis | $0.16 | 100 tracks × 20s avg |
| Elasticsearch | $0.00 | Free tier |
| Cloud Run (month 1) | $5-10 | ~30 new tracks |
| **Total hackathon** | **~$6** | One-time + month 1 |

## Troubleshooting

### "yt-dlp: command not found"
```bash
pip install --upgrade yt-dlp
```

### "ffmpeg: not found"
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### "Memory error during OpenL3"
Reduce batch size or use Cloud Run with 4GB+ memory:
```bash
# In cloudbuild.yaml, set:
memory: 4Gi
```

### Some videos fail to download
Normal! Geographic restrictions, age gates, etc. Script logs these to `*_failed.json`.

### Analysis seems slow
Expected! Real audio processing takes time:
- Downloading: 3-5s
- Librosa BPM: 2-3s
- Key detection: 1-2s
- OpenL3 embedding: 5-10s
- **Total: 12-23s per track**

This is what makes it impressive - **real AI audio analysis**!

## What's Next?

### After Hackathon
1. **Scale to 500 tracks** ($0.80 one-time)
2. **Add on-demand analysis** (analyze as users search)
3. **Implement caching** (never re-analyze same track)
4. **Add user feedback** (RLHF reranking)

### Production Alternatives
1. **Spotify API**: Free, instant, 100M+ tracks
2. **Licensed audio providers**: Azure Media Services, etc.
3. **Hybrid**: Spotify for metadata + YouTube for deep analysis

## Demo Tips

🎬 **Show the analysis happening live!**

"Watch as our AI analyzes the actual audio..."
- Downloads from YouTube
- Detects BPM using beat tracking
- Identifies musical key
- Generates 512-dim embedding
- All in 20 seconds!

This is WAY cooler than "we use Spotify API" 🚀

## Questions?

See `YOUTUBE_ANALYSIS.md` for detailed documentation.

---

**Ready? Let's go!** 🎵

```bash
bash setup_youtube_analysis.sh
```
