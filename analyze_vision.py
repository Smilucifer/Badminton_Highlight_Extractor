import cv2
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from ultralytics import YOLO

def process_video_chunk(args):
    """Worker function to process a specific chunk of the video."""
    video_path, start_frame, end_frame, skip_frames, batch_size = args
    
    # Initialize a localized model instance. YOLOv8 models are lightweight,
    # and giving each thread its own object avoids internal state collision,
    # whilePyTorch backend naturally handles concurrent GPU queuing.
    model = YOLO('yolov8n-pose.pt')
    
    cap = cv2.VideoCapture(video_path)
    # Seek to the designated start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    frame_metrics = []
    batch_frames = []
    batch_times = []
    
    frame_idx = start_frame
    
    while cap.isOpened() and frame_idx < end_frame:
        # Fast skip over frames we don't need
        for _ in range(skip_frames - 1):
            if not cap.grab():
                break
            frame_idx += 1
            if frame_idx >= end_frame:
                break
                
        if frame_idx >= end_frame:
            break
            
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_idx += 1
        
        batch_frames.append(frame)
        batch_times.append(frame_idx / fps)
        
        # Batch is full, offload to GPU
        if len(batch_frames) == batch_size:
            results = model(batch_frames, verbose=False, half=True)
            for i, result in enumerate(results):
                persons_data = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for j, box in enumerate(boxes):
                        if confs[j] < 0.5: continue
                        persons_data.append({
                            "id": j,
                            "cx": float((box[0] + box[2]) / 2),
                            "cy": float((box[1] + box[3]) / 2),
                            "height": float(box[3] - box[1])
                        })
                frame_metrics.append({"time": batch_times[i], "persons": persons_data})
            batch_frames.clear()
            batch_times.clear()
            
    # Process remainder frames
    if batch_frames:
        results = model(batch_frames, verbose=False, half=True)
        for i, result in enumerate(results):
            persons_data = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for j, box in enumerate(boxes):
                    if confs[j] < 0.5: continue
                    persons_data.append({
                        "id": j,
                        "cx": float((box[0] + box[2]) / 2),
                        "cy": float((box[1] + box[3]) / 2),
                        "height": float(box[3] - box[1])
                    })
            frame_metrics.append({"time": batch_times[i], "persons": persons_data})
            
    cap.release()
    return frame_metrics

def analyze_video_movement(video_path, output_json, batch_size=16, num_workers=4):
    print(f"Loading native YOLOv8-pose model (Multi-Threading setup)...")
    
    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    skip_frames = max(1, int(fps / 5))
    print(f"FPS: {fps}, Total frames: {total_frames}, Vid Stride: {skip_frames}, Inference Batch: {batch_size}")
    
    # Calculate video chunks
    chunk_size = total_frames // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = total_frames if i == num_workers - 1 else (i + 1) * chunk_size
        chunks.append((video_path, start, end, skip_frames, batch_size))
        
    print(f"\\n🚀 Splitting video into {num_workers} concurrent chunks for maximum resource utilization...")
    
    start_time = time.time()
    all_metrics = []
    
    # Spawn thread pool to process chunks concurrently
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {executor.submit(process_video_chunk, chunk): i for i, chunk in enumerate(chunks)}
        
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                metrics = future.result()
                all_metrics.extend(metrics)
                print(f"✅ Chunk {chunk_id + 1}/{num_workers} completed.")
            except Exception as exc:
                print(f"❌ Chunk {chunk_id + 1} generated an exception: {exc}")
                
    # Re-order the results because concurrent threads finish out of order
    all_metrics.sort(key=lambda x: x['time'])
    
    elapsed = time.time() - start_time
    real_fps = total_frames / elapsed if elapsed > 0 else 0
    print(f"\\n✨ Video processing complete in {elapsed:.2f}s!")
    print(f"🔥 Overall Engine Speed: {real_fps:.1f} Video FPS")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to {output_json}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        analyze_video_movement(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python analyze_vision.py <video.mp4> <output.json>")