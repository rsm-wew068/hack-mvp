#!/usr/bin/env python3
"""
FMA Dataset Parser and Embeddings Generator
Parses pre-computed FMA features and creates embeddings for our system
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import sys

# Paths
FMA_DIR = Path('/tmp/fma_metadata')
TRACKS_FILE = FMA_DIR / 'tracks.csv'
FEATURES_FILE = FMA_DIR / 'features.csv'
ECHONEST_FILE = FMA_DIR / 'echonest.csv'

OUTPUT_FILE = './fma_processed_tracks.json'
MAX_TRACKS = 1000  # Start with 1000 tracks


def load_fma_tracks():
    """Load FMA tracks metadata"""
    print("📚 Loading FMA tracks metadata...")
    
    # FMA tracks.csv has multi-level columns, skip first 2 rows
    tracks = pd.read_csv(TRACKS_FILE, index_col=0, header=[0, 1])
    
    print(f"✅ Loaded {len(tracks)} tracks")
    return tracks


def load_fma_features():
    """Load pre-computed audio features"""
    print("🎵 Loading pre-computed audio features...")
    
    # Features.csv also has multi-level columns
    features = pd.read_csv(FEATURES_FILE, index_col=0, header=[0, 1, 2])
    
    print(f"✅ Loaded features for {len(features)} tracks")
    return features


def load_fma_echonest():
    """Load Echo Nest audio analysis"""
    print("🎼 Loading Echo Nest audio analysis...")
    
    try:
        echonest = pd.read_csv(ECHONEST_FILE, index_col=0, header=[0, 1])
        print(f"✅ Loaded Echo Nest data for {len(echonest)} tracks")
        return echonest
    except Exception as e:
        print(f"⚠️  Echo Nest data not available: {e}")
        return None


def create_embeddings_from_mfcc(features_row):
    """
    Create 512-dimensional embedding from MFCC features
    MFCCs are the most important features for audio similarity
    """
    try:
        # Extract MFCC features (usually 20 coefficients x mean/std)
        # FMA features include: mfcc.mean.XX and mfcc.std.XX
        mfcc_features = []
        
        # Get all MFCC-related features
        for col in features_row.index:
            if 'mfcc' in str(col).lower():
                val = features_row[col]
                if pd.notna(val):
                    mfcc_features.append(float(val))
        
        # Get spectral features
        spectral_features = []
        for col in features_row.index:
            if any(x in str(col).lower() for x in ['spectral', 'chroma', 'tonnetz', 'zcr', 'rmse']):
                val = features_row[col]
                if pd.notna(val):
                    spectral_features.append(float(val))
        
        # Combine all features
        all_features = mfcc_features + spectral_features
        
        # Pad or truncate to 512 dimensions
        if len(all_features) < 512:
            # Pad with zeros
            embedding = all_features + [0.0] * (512 - len(all_features))
        else:
            # Truncate
            embedding = all_features[:512]
        
        return embedding
    
    except Exception as e:
        print(f"⚠️  Error creating embedding: {e}")
        # Return zero vector as fallback
        return [0.0] * 512


def extract_audio_features(echonest_row, features_row):
    """Extract BPM, key, and other audio features"""
    
    # Default values
    bpm = 120.0
    key = "C major"
    energy = 0.5
    danceability = 0.5
    valence = 0.5
    
    try:
        # Try to get BPM from Echo Nest
        if echonest_row is not None:
            if ('echonest', 'temporal_features', 'tempo') in echonest_row.index:
                tempo_val = echonest_row[('echonest', 'temporal_features', 'tempo')]
                if pd.notna(tempo_val):
                    bpm = float(tempo_val)
        
        # Try to get key from Echo Nest
        if echonest_row is not None:
            if ('echonest', 'temporal_features', 'key') in echonest_row.index:
                key_val = echonest_row[('echonest', 'temporal_features', 'key')]
                if pd.notna(key_val):
                    # Convert numeric key to note name
                    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    key_idx = int(key_val) % 12
                    key = f"{keys[key_idx]} major"  # Simplified
        
        # Energy and other features
        if echonest_row is not None:
            if ('echonest', 'audio_features', 'energy') in echonest_row.index:
                energy_val = echonest_row[('echonest', 'audio_features', 'energy')]
                if pd.notna(energy_val):
                    energy = float(energy_val)
            
            if ('echonest', 'audio_features', 'danceability') in echonest_row.index:
                dance_val = echonest_row[('echonest', 'audio_features', 'danceability')]
                if pd.notna(dance_val):
                    danceability = float(dance_val)
            
            if ('echonest', 'audio_features', 'valence') in echonest_row.index:
                val_val = echonest_row[('echonest', 'audio_features', 'valence')]
                if pd.notna(val_val):
                    valence = float(val_val)
    
    except Exception as e:
        print(f"⚠️  Error extracting audio features: {e}")
    
    return {
        'bpm': bpm,
        'key': key,
        'energy': energy,
        'danceability': danceability,
        'valence': valence
    }


def process_tracks(max_tracks=MAX_TRACKS):
    """Process FMA tracks and create our dataset"""
    
    print(f"\n🚀 Processing FMA dataset (max {max_tracks} tracks)...\n")
    
    # Load data
    tracks = load_fma_tracks()
    features = load_fma_features()
    echonest = load_fma_echonest()
    
    # Filter tracks that have both metadata and features
    common_ids = tracks.index.intersection(features.index)
    print(f"\n📊 Found {len(common_ids)} tracks with complete data")
    
    # Limit to max_tracks
    selected_ids = list(common_ids[:max_tracks])
    print(f"✂️  Processing {len(selected_ids)} tracks\n")
    
    processed_tracks = []
    
    for i, track_id in enumerate(selected_ids):
        try:
            # Get track metadata
            track_row = tracks.loc[track_id]
            features_row = features.loc[track_id]
            echonest_row = echonest.loc[track_id] if echonest is not None and track_id in echonest.index else None
            
            # Extract metadata
            try:
                title = str(track_row[('track', 'title')])
                artist = str(track_row[('artist', 'name')])
                album = str(track_row[('album', 'title')])
                genre = str(track_row[('track', 'genre_top')])
            except:
                # Fallback for different column structure
                title = f"Track {track_id}"
                artist = "Unknown Artist"
                album = "Unknown Album"
                genre = "Unknown"
            
            # Create embedding from MFCC features
            embedding = create_embeddings_from_mfcc(features_row)
            
            # Extract audio features
            audio_features = extract_audio_features(echonest_row, features_row)
            
            # Create track record
            track_record = {
                'track_id': f'fma_{track_id:06d}',
                'title': title,
                'artist': artist,
                'album': album,
                'genre': genre,
                'bpm': audio_features['bpm'],
                'key': audio_features['key'],
                'energy': audio_features['energy'],
                'danceability': audio_features['danceability'],
                'valence': audio_features['valence'],
                'openl3': embedding,  # Name it openl3 for compatibility
                'created_at': datetime.utcnow().isoformat()
            }
            
            processed_tracks.append(track_record)
            
            # Progress update
            if (i + 1) % 100 == 0:
                print(f"✅ Processed {i + 1}/{len(selected_ids)} tracks")
        
        except Exception as e:
            print(f"❌ Error processing track {track_id}: {e}")
            continue
    
    print(f"\n✨ Successfully processed {len(processed_tracks)} tracks")
    
    # Save to JSON
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(processed_tracks, f, indent=2)
    
    print(f"✅ Saved {len(processed_tracks)} tracks to {OUTPUT_FILE}")
    print(f"📦 File size: {Path(OUTPUT_FILE).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print sample track
    if processed_tracks:
        print("\n🎵 Sample track:")
        sample = processed_tracks[0]
        print(f"  Title: {sample['title']}")
        print(f"  Artist: {sample['artist']}")
        print(f"  Genre: {sample['genre']}")
        print(f"  BPM: {sample['bpm']:.1f}")
        print(f"  Key: {sample['key']}")
        print(f"  Embedding dimensions: {len(sample['openl3'])}")
    
    return processed_tracks


if __name__ == '__main__':
    max_tracks = MAX_TRACKS
    
    if len(sys.argv) > 1:
        try:
            max_tracks = int(sys.argv[1])
            print(f"🎯 Processing {max_tracks} tracks (from command line)")
        except:
            print(f"⚠️  Invalid argument, using default: {MAX_TRACKS}")
    
    tracks = process_tracks(max_tracks)
    
    print(f"\n✅ DONE! Processed {len(tracks)} tracks from FMA dataset")
    print(f"\nNext steps:")
    print(f"  1. python ingest_fma_to_bigquery.py")
    print(f"  2. python ingest_fma_to_elasticsearch.py")
    print(f"  3. ./start_services.sh")
