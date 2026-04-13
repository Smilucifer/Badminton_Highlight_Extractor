import cv2
import json
import numpy as np
from ultralytics import YOLO

def analyze_video_movement(video_path, output_json):
    print(f"Loading YOLOv8-pose model...")
    # Load YOLOv8 pose model (will download weights on first run)
    model = YOLO('yolov8n-pose.pt').to('cuda') 
    
    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(video_path)
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    frame_metrics = []
    
    frame_count = 0
    # Process every Nth frame to speed up MVP (e.g. process 5 frames per second)
    skip_frames = int(fps / 5) 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Skip frames for speed
        if frame_count % skip_frames != 0:
            continue
            
        current_time = frame_count / fps
        
        if frame_count % (int(fps) * 5) == 0:
            print(f"Processing time: {current_time:.2f}s / {total_frames/fps:.2f}s")
            
        # Run inference
        results = model(frame, verbose=False)
        
        # Analyze results for this frame
        persons_data = []
        if len(results) > 0 and results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            keypoints = results[0].keypoints.xy.cpu().numpy() # [N, 17, 2]
            confs = results[0].boxes.conf.cpu().numpy()
            
            for i, box in enumerate(boxes):
                # Filter low confidence detectons
                if confs[i] < 0.5:
                    continue
                    
                # We expect 2 people on court. Calculate their bounding box center
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                box_height = box[3] - box[1]
                
                persons_data.append({
                    "id": i,
                    "cx": float(center_x),
                    "cy": float(center_y),
                    "height": float(box_height)
                })
                
        # Store metrics for this second
        frame_metrics.append({
            "time": current_time,
            "persons": persons_data
        })
        
    cap.release()
    print("Video processing complete.")
    
    # Process metrics to find "still" moments and "active" moments
    # In badminton, before serving, players are relatively still.
    print("Analyzing movement patterns...")
    
    # We will look for moments where player bounding box centers don't move much over 1-2 seconds
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(frame_metrics, f, indent=4)
    print(f"Metrics saved to {output_json}")

if __name__ == "__main__":
    video_file = "/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp/clip_3min.mp4"
    output_file = "/mnt/d/ClaudeWorkspace/Code/badminton_audio_mvp/vision_metrics.json"
    analyze_video_movement(video_file, output_file)
