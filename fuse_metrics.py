import json
import os
import math

def calculate_movement(vision_data, start_time, end_time):
    # Filter vision frames within the rally time
    rally_frames = [f for f in vision_data if start_time <= f['time'] <= end_time]
    
    if not rally_frames:
        return 0, 0
        
    total_movement = 0
    frame_count = len(rally_frames)
    
    # Simple approach: sum the displacement of the closest matching ID between consecutive frames
    # Since IDs might not be mapped perfectly by raw YOLO, we compute naive frame-to-frame gross motion
    
    for i in range(1, frame_count):
        prev_persons = rally_frames[i-1]['persons']
        curr_persons = rally_frames[i]['persons']
        
        frame_motion = 0
        # For each person in current frame, find closest in prev frame to calculate speed
        for cp in curr_persons:
            min_dist = float('inf')
            for pp in prev_persons:
                dist = math.hypot(cp['cx'] - pp['cx'], cp['cy'] - pp['cy'])
                if dist < min_dist:
                    min_dist = dist
            
            # Cap unreasonable movement (e.g. ID switch across court)
            if min_dist != float('inf') and min_dist < 500: 
                frame_motion += min_dist
                
        total_movement += frame_motion
        
    avg_movement_per_sec = total_movement / (end_time - start_time) if end_time > start_time else 0
    return total_movement, avg_movement_per_sec

def main():
    base_dir = '/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp'
    rallies_path = os.path.join(base_dir, 'rallies.json')
    vision_path = os.path.join(base_dir, 'vision_metrics.json')
    output_path = os.path.join(base_dir, 'high_value_rallies.json')
    
    print("Loading audio rallies...")
    with open(rallies_path, 'r') as f:
        rallies = json.load(f)
        
    print("Loading vision metrics...")
    with open(vision_path, 'r') as f:
        vision_data = json.load(f)
        
    print(f"Fusing data for {len(rallies)} rallies...")
    
    scored_rallies = []
    
    for rally in rallies:
        total_mov, avg_mov = calculate_movement(vision_data, rally['start_time'], rally['end_time'])
        
        # We can construct a "highlight score". 
        # Factors: Duration (longer is usually better up to a point), Total movement
        
        # Normalized naive score: 
        # Multiplier based on audio (duration) and vision (avg_movement)
        
        # Penalty if it's too short (not a real rally)
        duration = rally['duration']
        
        score = (duration * 0.4) + (avg_mov * 0.05)
        
        rally['vision_total_movement'] = round(total_mov, 2)
        rally['vision_avg_movement_per_sec'] = round(avg_mov, 2)
        rally['highlight_score'] = round(score, 2)
        
        scored_rallies.append(rally)
        
    # Sort by highlight score
    scored_rallies.sort(key=lambda x: x['highlight_score'], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scored_rallies, f, indent=4)
        
    print(f"Finished! Top 3 Rallies:")
    for i, r in enumerate(scored_rallies[:3]):
        print(f"#{i+1}: Score: {r['highlight_score']} | Time: {r['start_time']}s - {r['end_time']}s | Dur: {r['duration']}s | Avg Mov: {r['vision_avg_movement_per_sec']}")

if __name__ == "__main__":
    main()
