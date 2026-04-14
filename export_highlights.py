import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def _run_ffmpeg(start, end, video_path, out_file, index):
    """Worker function to cut a single highlight clip using ffmpeg."""
    print(f"[{index}] Triggered cut ({start:.1f}s - {end:.1f}s)...")
    cmd = [
        'ffmpeg', '-y', 
        '-ss', str(start), 
        '-i', video_path, 
        '-to', str(end - start), # when -i is after -ss, -to represents duration
        '-c:v', 'copy', '-c:a', 'copy', 
        out_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return index, out_file

def export_highlights(video_path, rallies_path, output_dir, max_workers=5):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading scored rallies from {rallies_path}...")
    with open(rallies_path, 'r') as f:
        rallies = json.load(f)
        
    # Only keep rallies with score >= 80
    valid_rallies = [r for r in rallies if r.get('highlight_score', 0) >= 80]
    print(f"Found {len(valid_rallies)} highlight candidates.")
    
    # Ensure they are sorted chronologically
    valid_rallies.sort(key=lambda x: x['start_time'])
    
    tasks = []
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Pre-calculate all cuts
    for i, rally in enumerate(valid_rallies):
        pad_front = 1.5
        pad_back = 2.0
        
        if i < len(valid_rallies) - 1:
            next_serve_time = valid_rallies[i+1]['start_time']
            available_gap = next_serve_time - rally['end_time'] - 0.2
            if available_gap < 0:
                available_gap = 0
            pad_back = min(pad_back, available_gap)

        start = max(0, rally['start_time'] - pad_front)
        end = rally['end_time'] + pad_back
        
        out_file = os.path.join(output_dir, f"{vid_name}_highlight_{i+1}_score_{rally['highlight_score']:.1f}.mp4")
        
        tasks.append((start, end, video_path, out_file, i + 1))
        
    # Process cuts concurrently
    if tasks:
        print(f"\\n✂️ Spawning {max_workers} threads to process the cuts in parallel...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(_run_ffmpeg, t[0], t[1], t[2], t[3], t[4]): t for t in tasks}
            
            # Wait for them to complete
            for future in as_completed(future_to_task):
                try:
                    index, saved_file = future.result()
                    print(f"✅ Highlight #{index} saved: {os.path.basename(saved_file)}")
                except Exception as exc:
                    print(f"❌ Highlight task generated an exception: {exc}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        export_highlights(sys.argv[1], sys.argv[2], sys.argv[3])