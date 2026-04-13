import json
import os
import subprocess

def main():
    base_dir = '/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp'
    rallies_path = os.path.join(base_dir, 'high_value_rallies.json')
    video_path = os.path.join(base_dir, 'clip_3min.mp4')
    output_dir = os.path.join(base_dir, 'highlights_3min')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading scored rallies...")
    with open(rallies_path, 'r') as f:
        rallies = json.load(f)
        
    # Only keep rallies with score >= 80
    valid_rallies = [r for r in rallies if r.get('highlight_score', 0) >= 80]
    
    print(f"Found {len(valid_rallies)} highlight candidates within the video duration.")
    
    # Ensure they are sorted chronologically
    valid_rallies.sort(key=lambda x: x['start_time'])
    
    # Export all chronological highlights that match criteria
    for i, rally in enumerate(valid_rallies):
        # Adaptive padding to avoid overlapping the visual action of the next/preview rally
        pad_front = 1.5
        pad_back = 2.0
        
        # 前置缓冲：允许重叠，但最多只倒推 1.5 秒
        # 此处不严格限制 pad_front，允许前一个回合结束的画面出现在这里的缓冲中（符合“允许重叠”特性）
            
        if i < len(valid_rallies) - 1:
            next_serve_time = valid_rallies[i+1]['start_time']
            # 后置缓冲：绝对不能把下一个球的“发球瞬间”(next_serve_time) 给剪进来
            # 我们限制后置缓冲最多只能延伸到下一个发球前 0.2 秒
            available_gap = next_serve_time - rally['end_time'] - 0.2
            if available_gap < 0:
                available_gap = 0
            pad_back = min(pad_back, available_gap)

        start = max(0, rally['start_time'] - pad_front)
        end = rally['end_time'] + pad_back
        
        out_file = os.path.join(output_dir, f"highlight_{i+1}_score_{rally['highlight_score']:.1f}.mp4")
        
        print(f"Cutting highlight #{i+1} ({start}s - {end}s)...")
        # Use simple ffmpeg stream copy to be fast
        cmd = [
            'ffmpeg', '-y', '-i', video_path, 
            '-ss', str(start), '-to', str(end), 
            '-c:v', 'copy', '-c:a', 'copy', 
            out_file
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
