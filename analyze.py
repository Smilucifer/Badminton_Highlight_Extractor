import librosa
import numpy as np
from scipy.signal import find_peaks
import json

def analyze_audio(audio_path, output_json):
    print(f"Loading audio from {audio_path}...")
    # Load audio, resample to 16kHz for speed (we already did this in ffmpeg)
    y, sr = librosa.load(audio_path, sr=None)
    
    print("Calculating energy envelope...")
    # Compute the short-time energy of the audio to find sharp sounds (hits)
    # We use a small hop length to get good time resolution
    hop_length = 512
    energy = np.array([sum(abs(y[i:i+hop_length]**2)) for i in range(0, len(y), hop_length)])
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
    
    # Normalize energy
    energy = energy / np.max(energy)
    
    print("Detecting peaks (badminton hits)...")
    # Threshold: adjust based on background noise. 0.05 is a starting guess.
    # Distance: at least 0.2 seconds between hits to avoid double-counting the same hit.
    peaks, properties = find_peaks(energy, height=0.02, distance=int(sr * 0.2 / hop_length))
    hit_times = times[peaks]
    
    print(f"Found {len(hit_times)} potential hits.")
    
    print("Clustering hits into rallies...")
    # If the gap between two hits is more than 3.5 seconds, we consider it a new rally.
    rally_gap_threshold = 3.5
    
    rallies = []
    if len(hit_times) > 0:
        current_rally_start = hit_times[0]
        last_hit = hit_times[0]
        
        for hit in hit_times[1:]:
            if hit - last_hit > rally_gap_threshold:
                # Consider it a valid rally if it lasted at least 1.5 seconds (typically >1 hit)
                if last_hit - current_rally_start > 1.5:
                    rallies.append({
                        "start_time": round(current_rally_start, 2),
                        "end_time": round(last_hit + 1.0, 2), # Pad 1 second at the end
                        "duration": round((last_hit + 1.0) - current_rally_start, 2)
                    })
                current_rally_start = hit
            last_hit = hit
            
        # Add the last rally
        if last_hit - current_rally_start > 1.5:
            rallies.append({
                "start_time": round(current_rally_start, 2),
                "end_time": round(last_hit + 1.0, 2),
                "duration": round((last_hit + 1.0) - current_rally_start, 2)
            })
            
    print(f"Extracted {len(rallies)} rallies.")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(rallies, f, indent=4)
        
    return rallies

if __name__ == "__main__":
    audio_file = "/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp/track.wav"
    output_file = "/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp/rallies.json"
    analyze_audio(audio_file, output_file)
