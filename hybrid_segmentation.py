import json
import math
import os

import librosa
import numpy as np
from scipy.signal import find_peaks

from config import AppConfig


def load_vision_data(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def compute_vision_velocities(vision_data: list[dict]) -> list[tuple[float, float]]:
    velocities = []
    for index in range(1, len(vision_data)):
        prev = vision_data[index - 1]
        curr = vision_data[index]
        dt = curr["time"] - prev["time"]
        if dt <= 0:
            continue

        frame_motion = 0.0
        for curr_person in curr["persons"]:
            min_dist = float("inf")
            for prev_person in prev["persons"]:
                dist = math.hypot(
                    curr_person["cx"] - prev_person["cx"],
                    curr_person["cy"] - prev_person["cy"],
                )
                if dist < min_dist:
                    min_dist = dist
            if min_dist != float("inf") and min_dist < 400:
                frame_motion += min_dist

        velocities.append((curr["time"], frame_motion / dt))
    return velocities


def count_intense_frames(
    velocities: list[tuple[float, float]],
    start_t: float,
    end_t: float,
    intense_threshold: float,
) -> float:
    window_values = [value for time_value, value in velocities if start_t <= time_value <= end_t]
    if not window_values:
        return 0.0
    intense_count = sum(1 for value in window_values if value > intense_threshold)
    return intense_count / len(window_values)


def segment_video(
    audio_path: str,
    vision_path: str,
    output_rallies_path: str,
    config: AppConfig,
) -> dict:
    print(f"Segmenting based on {audio_path} and {vision_path}")
    vision_data = load_vision_data(vision_path)
    velocities = compute_vision_velocities(vision_data)
    all_velocity_values = [value for _, value in velocities]

    stats = {
        "audio_hits": 0,
        "valid_hits": 0,
        "rallies": 0,
        "discarded_short_rallies": 0,
        "intense_threshold": None,
        "valid_hit_times": [],
    }

    output_dir = os.path.dirname(output_rallies_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not all_velocity_values:
        with open(output_rallies_path, "w", encoding="utf-8") as file:
            json.dump([], file, indent=4)
        stats["warning"] = "No motion velocities found in vision data."
        print("No velocities found in vision data.")
        return stats

    intense_threshold = float(np.percentile(all_velocity_values, config.fusion.motion_percentile))
    stats["intense_threshold"] = intense_threshold
    print(f"Intense Threshold: {intense_threshold:.1f}")

    y, sr = librosa.load(audio_path, sr=config.audio.sample_rate)
    hop_length = config.audio.hop_length
    energy = np.array(
        [
            np.sum(np.abs(y[index:index + hop_length] ** 2))
            for index in range(0, len(y), hop_length)
        ]
    )
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
    if np.max(energy) > 0:
        energy = energy / np.max(energy)

    min_peak_distance = max(1, int(sr * config.audio.min_peak_distance_sec / hop_length))
    peaks, _ = find_peaks(
        energy,
        height=config.audio.peak_height,
        distance=min_peak_distance,
    )
    stats["energy_points"] = [
        (float(time_value), float(energy_value))
        for time_value, energy_value in zip(times, energy)
    ]
    audio_hits = times[peaks]
    stats["audio_hits"] = int(len(audio_hits))

    valid_hits = []
    for hit in audio_hits:
        intense_ratio = count_intense_frames(
            velocities,
            hit - config.fusion.hit_window_before,
            hit + config.fusion.hit_window_after,
            intense_threshold,
        )
        if intense_ratio > config.fusion.min_intense_ratio:
            valid_hits.append(hit)

    stats["valid_hits"] = int(len(valid_hits))
    stats["valid_hit_times"] = [round(hit, 3) for hit in valid_hits]
    stats["velocity_points"] = [(float(time_value), float(value)) for time_value, value in velocities]
    print(f"Verified hits: {len(valid_hits)}")

    rallies = []
    if valid_hits:
        current_rally_hits = [valid_hits[0]]
        for hit in valid_hits[1:]:
            if hit - current_rally_hits[-1] > config.fusion.rally_gap_threshold:
                duration = current_rally_hits[-1] - current_rally_hits[0]
                if duration > config.fusion.min_rally_duration:
                    rallies.append(
                        {
                            "start_time": current_rally_hits[0],
                            "end_time": current_rally_hits[-1],
                            "duration": duration,
                            "valid_hits": current_rally_hits.copy(),
                        }
                    )
                else:
                    stats["discarded_short_rallies"] += 1
                current_rally_hits = [hit]
            else:
                current_rally_hits.append(hit)

        duration = current_rally_hits[-1] - current_rally_hits[0]
        if duration > config.fusion.min_rally_duration:
            rallies.append(
                {
                    "start_time": current_rally_hits[0],
                    "end_time": current_rally_hits[-1],
                    "duration": duration,
                    "valid_hits": current_rally_hits.copy(),
                }
            )
        else:
            stats["discarded_short_rallies"] += 1

    output_rallies = []
    for index, rally in enumerate(rallies, start=1):
        raw_start_time = round(rally["start_time"], 2)
        raw_end_time = round(rally["end_time"], 2)
        duration = round(rally["duration"], 2)
        rally_payload = {
            "raw_start_time": raw_start_time,
            "raw_end_time": raw_end_time,
            "start_time": raw_start_time,
            "end_time": raw_end_time,
            "duration": duration,
            "highlight_score": round(duration * 10, 2),
            "vision_avg_movement_per_sec": 100,
            "valid_hits": [round(hit, 3) for hit in rally.get("valid_hits", [])],
        }
        print(
            f"Rally #{index}: {rally_payload['start_time']:05.2f}s to {rally_payload['end_time']:05.2f}s | "
            f"Dur: {rally_payload['duration']:04.2f}s"
        )
        output_rallies.append(rally_payload)

    with open(output_rallies_path, "w", encoding="utf-8") as file:
        json.dump(output_rallies, file, indent=4)

    stats["rallies"] = len(output_rallies)
    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        raise SystemExit(
            "segment_video now requires an AppConfig instance and should be called from main.py"
        )
