import json
import os
import math
import librosa
import numpy as np
from scipy.signal import find_peaks

def load_vision_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def compute_vision_velocities(vision_data):
    velocities = []
    for i in range(1, len(vision_data)):
        prev = vision_data[i-1]
        curr = vision_data[i]
        dt = curr['time'] - prev['time']
        if dt <= 0: continue
        
        frame_motion = 0
        for cp in curr['persons']:
            min_dist = float('inf')
            for pp in prev['persons']:
                dist = math.hypot(cp['cx'] - pp['cx'], cp['cy'] - pp['cy'])
                if dist < min_dist:
                    min_dist = dist
            if min_dist != float('inf') and min_dist < 400:
                frame_motion += min_dist
                
        velocities.append((curr['time'], frame_motion / dt))
    return velocities

def count_intense_frames(velocities, start_t, end_t, intense_threshold):
    v = [vel[1] for vel in velocities if start_t <= vel[0] <= end_t]
    if not v: return 0
    return sum(1 for val in v if val > intense_threshold) / len(v)

def segment_video(audio_path, vision_path, output_rallies_path):
    print(f"Segmenting based on {audio_path} and {vision_path}")
    velocities = compute_vision_velocities(load_vision_data(vision_path))
    all_vels = [v[1] for v in velocities]
    
    # 低剧烈度，只要不散步就行，毕竟有单打的时候活动范围没那么大
    if not all_vels:
        print("No velocities found in vision data!")
        return
        
    intense_threshold = np.percentile(all_vels, 50) 
    print(f"Intense Threshold: {intense_threshold:.1f}")

    y, sr = librosa.load(audio_path, sr=None)
    hop_length = 512
    energy = np.array([sum(abs(y[i:i+hop_length]**2)) for i in range(0, len(y), hop_length)])
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    peaks, _ = find_peaks(energy, height=0.015, distance=int(sr * 0.2 / hop_length))
    audio_hits = times[peaks]

    valid_hits = []
    for hit in audio_hits:
        intense_ratio = count_intense_frames(velocities, hit-0.2, hit + 1.2, intense_threshold)
        # 宽容的打球特征，只要发球前后稍微动了动
        if intense_ratio > 0.05:
            valid_hits.append(hit)
            
    print(f"Verified hits: {len(valid_hits)}")
    
    # 核心：2.5秒无击球，判定为死球。
    rally_gap_threshold = 2.5
    
    rallies = []
    if len(valid_hits) > 0:
        current_rally_hits = [valid_hits[0]]
        
        for hit in valid_hits[1:]:
            if hit - current_rally_hits[-1] > rally_gap_threshold:
                if current_rally_hits[-1] - current_rally_hits[0] > 1.0:
                    rallies.append({
                        "start_time": current_rally_hits[0],
                        "end_time": current_rally_hits[-1],
                        "duration": current_rally_hits[-1] - current_rally_hits[0]
                    })
                current_rally_hits = [hit]
            else:
                current_rally_hits.append(hit)
                
        if current_rally_hits[-1] - current_rally_hits[0] > 1.0:
            rallies.append({
                "start_time": current_rally_hits[0],
                "end_time": current_rally_hits[-1],
                "duration": current_rally_hits[-1] - current_rally_hits[0]
            })

    output_rallies = []
    for i, r in enumerate(rallies):
        r['start_time'] = round(r['start_time'], 2)
        r['end_time'] = round(r['end_time'], 2)
        r['duration'] = round(r['duration'], 2)
        r['highlight_score'] = r['duration'] * 10
        r['vision_avg_movement_per_sec'] = 100 
        print(f"Rally #{i+1}: {r['start_time']:05.2f}s to {r['end_time']:05.2f}s | Dur: {r['duration']:04.2f}s")
        output_rallies.append(r)
        
    with open(output_rallies_path, 'w') as f:
        json.dump(output_rallies, f, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        segment_video(sys.argv[1], sys.argv[2], sys.argv[3])
