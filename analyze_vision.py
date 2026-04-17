import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from ultralytics import YOLO

from config import AppConfig


def process_video_chunk(args) -> tuple[list[dict], dict]:
    video_path, start_frame, end_frame, config = args

    if not os.path.exists(config.vision.model_path):
        raise FileNotFoundError(f"Model file not found: {config.vision.model_path}")

    model = YOLO(config.vision.model_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_stride = max(1, int(fps / config.vision.sample_fps))

    frame_metrics = []
    batch_frames = []
    batch_times = []
    sampled_frames = 0
    frames_with_person = 0
    detections = 0
    frame_idx = start_frame

    while cap.isOpened() and frame_idx < end_frame:
        for _ in range(sample_stride - 1):
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
        sampled_frames += 1
        batch_frames.append(frame)
        batch_times.append(frame_idx / fps)

        if len(batch_frames) == config.vision.batch_size:
            results = model(
                batch_frames,
                verbose=False,
                half=config.vision.use_half_precision,
            )
            for index, result in enumerate(results):
                persons_data = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for box_index, box in enumerate(boxes):
                        if confs[box_index] < config.vision.confidence_threshold:
                            continue
                        persons_data.append(
                            {
                                "id": box_index,
                                "cx": float((box[0] + box[2]) / 2),
                                "cy": float((box[1] + box[3]) / 2),
                                "height": float(box[3] - box[1]),
                            }
                        )
                if persons_data:
                    frames_with_person += 1
                    detections += len(persons_data)
                frame_metrics.append({"time": batch_times[index], "persons": persons_data})
            batch_frames.clear()
            batch_times.clear()

    if batch_frames:
        results = model(batch_frames, verbose=False, half=True)
        for index, result in enumerate(results):
            persons_data = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for box_index, box in enumerate(boxes):
                    if confs[box_index] < config.vision.confidence_threshold:
                        continue
                    persons_data.append(
                        {
                            "id": box_index,
                            "cx": float((box[0] + box[2]) / 2),
                            "cy": float((box[1] + box[3]) / 2),
                            "height": float(box[3] - box[1]),
                        }
                    )
            if persons_data:
                frames_with_person += 1
                detections += len(persons_data)
            frame_metrics.append({"time": batch_times[index], "persons": persons_data})

    cap.release()
    return frame_metrics, {
        "sampled_frames": sampled_frames,
        "frames_with_person": frames_with_person,
        "detections": detections,
    }


def analyze_video_movement(
    video_path: str,
    output_json: str,
    config: AppConfig,
) -> dict:
    print("Loading YOLOv8 pose model...")
    print(f"Opening video {video_path}...")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(config.vision.model_path):
        raise FileNotFoundError(f"Model file not found: {config.vision.model_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    num_workers = max(1, config.vision.num_workers)
    sample_stride = max(1, int(fps / config.vision.sample_fps))
    chunk_size = total_frames // num_workers if num_workers > 0 else total_frames
    chunks = []
    for index in range(num_workers):
        start = index * chunk_size
        end = total_frames if index == num_workers - 1 else (index + 1) * chunk_size
        chunks.append((video_path, start, end, config))

    print(
        f"FPS: {fps}, Total frames: {total_frames}, Stride: {sample_stride}, "
        f"Batch: {config.vision.batch_size}, Workers: {num_workers}"
    )

    start_time = time.time()
    all_metrics = []
    stats = {
        "fps": float(fps),
        "total_frames": total_frames,
        "sample_stride": sample_stride,
        "sampled_frames": 0,
        "frames_with_person": 0,
        "detections": 0,
        "elapsed_sec": 0.0,
    }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(process_video_chunk, chunk): chunk
            for chunk in chunks
        }
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                metrics, chunk_stats = future.result()
                all_metrics.extend(metrics)
                stats["sampled_frames"] += chunk_stats["sampled_frames"]
                stats["frames_with_person"] += chunk_stats["frames_with_person"]
                stats["detections"] += chunk_stats["detections"]
                print(f"✅ Chunk completed: {chunk[1]}-{chunk[2]}")
            except Exception as exc:
                raise RuntimeError(
                    f"Vision analysis failed for {os.path.basename(video_path)} at chunk {chunk[1]}-{chunk[2]}: {exc}"
                ) from exc

    all_metrics.sort(key=lambda item: item["time"])
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=4)

    elapsed = time.time() - start_time
    stats["elapsed_sec"] = elapsed
    print(f"Metrics saved to {output_json}")
    print(
        f"Vision summary: sampled_frames={stats['sampled_frames']}, "
        f"frames_with_person={stats['frames_with_person']}, detections={stats['detections']}, "
        f"elapsed={elapsed:.2f}s"
    )
    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        raise SystemExit(
            "analyze_video_movement now requires an AppConfig instance and should be called from main.py"
        )
    else:
        print("Usage: python analyze_vision.py <video.mp4> <output.json>")
