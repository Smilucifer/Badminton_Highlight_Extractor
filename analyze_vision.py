import cv2
import json
import numpy as np
from ultralytics import YOLO

def analyze_video_movement(video_path, output_json):
    print(f"Loading YOLOv8-pose model...")
    # Load YOLOv8 pose model (will download weights on first run)
    # model = YOLO('yolov8n-pose.pt').to('cuda')
    model = YOLO('yolov8n-pose.engine')

    print(f"Opening video {video_path}...")
    
    # We still need cv2 just to probe the video metadata quickly
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # default fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Process every Nth frame to speed up MVP (e.g. process 5 frames per second)
    skip_frames = max(1, int(fps / 5)) 
    print(f"Vid_Stride set to: {skip_frames}")

    frame_metrics = []
    
    # Let Ultralytics handle decoding, striding, and batching natively.
    # stream=True returns a generator reducing memory overhead.
    # vid_stride applies our skip logic internally at the decoding layer.
    # half=True triggers FP16 mixed precision for massive speedups on RTX GPUs.
    # batch isn't strictly needed for stream mode since ultralytics batches internally when configured right,
    # but providing stream=True avoids the memory bloat of loading the whole video tensor.
    results_generator = model(
        video_path, 
        stream=True, 
        vid_stride=skip_frames, 
        verbose=False, 
        half=True
    )
    
    for i, result in enumerate(results_generator):
        frame_idx = i * skip_frames
        current_time = frame_idx / fps
        
        # Print roughly every 5 real video seconds
        if (frame_idx % (int(fps) * 5)) < skip_frames:
            print(f"Processing time: {current_time:.2f}s / {total_frames/fps:.2f}s")
            
        # Analyze results for this frame
        persons_data = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for j, box in enumerate(boxes):
                # Filter low confidence detectons
                if confs[j] < 0.5:
                    continue
                    
                # We expect 2 people on court. Calculate their bounding box center
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                box_height = box[3] - box[1]
                
                persons_data.append({
                    "id": j,
                    "cx": float(center_x),
                    "cy": float(center_y),
                    "height": float(box_height)
                })
                
        # Store metrics for this frame
        frame_metrics.append({
            "time": current_time,
            "persons": persons_data
        })
        
    print("Video processing complete.")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(frame_metrics, f, indent=4)
    print(f"Metrics saved to {output_json}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        analyze_video_movement(sys.argv[1], sys.argv[2])
