import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from boundary_refinement import refine_boundaries
from config import AppConfig, get_default_config, merge_config_overrides


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Please install FFmpeg and ensure it is available in the Windows terminal."
        )


def ensure_model_exists(config: AppConfig) -> None:
    if not os.path.exists(config.vision.model_path):
        raise FileNotFoundError(
            f"Model file not found: {config.vision.model_path}. Place yolov8n-pose.pt in the project root or update config."
        )


def extract_audio(video_path: str, audio_path: str) -> None:
    print(f"Extracting audio from {os.path.basename(video_path)}...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Audio extraction failed for {os.path.basename(video_path)}: {result.stderr.strip()}"
        )


def load_config(config_path: str | None) -> AppConfig:
    config = get_default_config()
    if not config_path:
        return config

    with open(config_path, "r", encoding="utf-8") as file:
        overrides = json.load(file)
    return merge_config_overrides(config, overrides)


def process_directory(target_dir: str, config: AppConfig) -> None:
    target_dir = os.path.abspath(target_dir)
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a directory.")
        sys.exit(1)

    ensure_ffmpeg_available()
    ensure_model_exists(config)

    from analyze_vision import analyze_video_movement
    from export_highlights import export_highlights
    from hybrid_segmentation import segment_video

    print(f"Scanning {target_dir} for video files...")

    valid_exts = {".mp4", ".mkv", ".avi", ".mov"}
    videos = []

    for file_name in os.listdir(target_dir):
        ext = os.path.splitext(file_name)[1].lower()
        if ext in valid_exts:
            videos.append(os.path.join(target_dir, file_name))

    if not videos:
        print("No videos found.")
        return

    print(f"Found {len(videos)} videos. Processing...")

    highlights_dir = os.path.join(target_dir, "highlights")
    os.makedirs(highlights_dir, exist_ok=True)

    for index, video_path in enumerate(videos, start=1):
        video_name = os.path.basename(video_path)
        print(f"\n[{index}/{len(videos)}] Processing: {video_name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "temp_audio.wav")
            vision_path = os.path.join(temp_dir, "temp_vision.json")
            rallies_path = os.path.join(temp_dir, "temp_rallies.json")

            try:
                print("  - Extracting audio...")
                extract_audio(video_path, audio_path)

                print("  - Running vision analysis...")
                vision_stats = analyze_video_movement(video_path, vision_path, config=config)

                print("  - Running hybrid segmentation...")
                segmentation_stats = segment_video(audio_path, vision_path, rallies_path, config=config)

                with open(rallies_path, "r", encoding="utf-8") as file:
                    rough_rallies = json.load(file)

                refined_rallies = refine_boundaries(
                    rough_rallies,
                    segmentation_stats.get("velocity_points", []),
                    segmentation_stats.get("energy_points", []),
                    config,
                )

                with open(rallies_path, "w", encoding="utf-8") as file:
                    json.dump(refined_rallies, file, indent=4)

                signal_refined_count = sum(
                    1 for rally in refined_rallies if rally.get("refinement_status") != "fallback_both"
                )
                full_signal_refined_count = sum(
                    1 for rally in refined_rallies if rally.get("refinement_status") == "signal_matched_both"
                )
                partial_signal_refined_count = sum(
                    1
                    for rally in refined_rallies
                    if rally.get("refinement_status") in {"signal_matched_start_only", "signal_matched_end_only"}
                )
                fallback_refined_count = sum(
                    1 for rally in refined_rallies if rally.get("refinement_status") == "fallback_both"
                )
                next_serve_clamped_count = sum(
                    1 for rally in refined_rallies if rally.get("boundary_reason_end") == "next_serve_protection"
                )
                start_fallback_count = sum(
                    1 for rally in refined_rallies if rally.get("boundary_reason_start") == "fallback_front_pad"
                )
                end_fallback_count = sum(
                    1 for rally in refined_rallies if rally.get("boundary_reason_end") == "fallback_back_pad"
                )
                full_fallback_count = sum(
                    1 for rally in refined_rallies if rally.get("refinement_status") == "fallback_both"
                )
                unchanged_count = sum(
                    1
                    for rally in refined_rallies
                    if abs(rally.get("raw_start_time", 0.0) - rally.get("start_time", 0.0)) < 0.05
                    and abs(rally.get("raw_end_time", 0.0) - rally.get("end_time", 0.0)) < 0.05
                )
                total_refined = max(1, len(refined_rallies))
                start_fallback_rate = round(start_fallback_count / total_refined, 3)
                end_fallback_rate = round(end_fallback_count / total_refined, 3)
                full_fallback_rate = round(full_fallback_count / total_refined, 3)
                print(
                    "  - Refinement metrics:"
                    f" signal_refined={signal_refined_count}"
                    f", full_signal={full_signal_refined_count}"
                    f", partial_signal={partial_signal_refined_count}"
                    f", fallback={fallback_refined_count}"
                    f", next_serve_clamped={next_serve_clamped_count}"
                    f", unchanged={unchanged_count}"
                    f", start_fallback_rate={start_fallback_rate}"
                    f", end_fallback_rate={end_fallback_rate}"
                    f", full_fallback_rate={full_fallback_rate}"
                )

                print("  - Exporting highlights...")
                export_stats = export_highlights(video_path, rallies_path, highlights_dir, config=config)

                print(
                    "  - Summary:"
                    f" sampled_frames={vision_stats.get('sampled_frames', 0)}"
                    f", frames_with_person={vision_stats.get('frames_with_person', 0)}"
                    f", audio_hits={segmentation_stats.get('audio_hits', 0)}"
                    f", valid_hits={segmentation_stats.get('valid_hits', 0)}"
                    f", rallies={segmentation_stats.get('rallies', 0)}"
                    f", exported={export_stats.get('exported', 0)}"
                    f", failed_exports={export_stats.get('failed_exports', 0)}"
                )
            except Exception as exc:
                print(f"  - Failed on {video_name}: {exc}")

    print(f"\nAll done! Highlights saved to: {highlights_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto cut badminton highlights.")
    parser.add_argument("path", help="Directory containing the target videos")
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Optional path to a JSON file with config overrides",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    process_directory(args.path, config)
