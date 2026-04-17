import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import AppConfig


def _run_ffmpeg(
    start: float,
    end: float,
    video_path: str,
    out_file: str,
    index: int,
) -> tuple[int, str, bool, str | None]:
    print(f"[{index}] Triggered cut ({start:.1f}s - {end:.1f}s)...")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        video_path,
        "-to",
        str(end - start),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        out_file,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        error_message = (
            f"ffmpeg export failed for highlight #{index} ({start:.1f}s - {end:.1f}s) "
            f"from {os.path.basename(video_path)}: {result.stderr.strip()}"
        )
        return index, out_file, False, error_message
    return index, out_file, True, None


def export_highlights(
    video_path: str,
    rallies_path: str,
    output_dir: str,
    config: AppConfig,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading scored rallies from {rallies_path}...")
    with open(rallies_path, "r", encoding="utf-8") as file:
        rallies = json.load(file)

    valid_rallies = [
        rally for rally in rallies if rally.get("highlight_score", 0) >= config.export.min_highlight_score
    ]
    valid_rallies.sort(key=lambda item: item["start_time"])

    stats = {
        "total_rallies": len(rallies),
        "eligible_rallies": len(valid_rallies),
        "exported": 0,
        "failed_exports": 0,
    }

    print(f"Found {len(valid_rallies)} highlight candidates.")
    if not valid_rallies:
        print("No highlights met export threshold.")
        return stats

    tasks = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    for index, rally in enumerate(valid_rallies):
        start = float(rally.get("start_time", rally.get("raw_start_time", 0.0)))
        end = float(rally.get("end_time", rally.get("raw_end_time", start)))

        if index < len(valid_rallies) - 1:
            next_serve_time = float(
                valid_rallies[index + 1].get(
                    "raw_start_time",
                    valid_rallies[index + 1].get("start_time", end),
                )
            )
            end = min(end, next_serve_time - config.boundary.post_hit_guard)

        start = max(0.0, start)
        end = max(start, end)

        out_file = os.path.join(
            output_dir,
            f"{video_name}_highlight_{index + 1}_score_{rally['highlight_score']:.1f}.mp4",
        )
        tasks.append((start, end, video_path, out_file, index + 1))

    print(f"\n✂️ Spawning {config.export.max_workers} threads to process cuts in parallel...")
    with ThreadPoolExecutor(max_workers=config.export.max_workers) as executor:
        future_to_task = {
            executor.submit(_run_ffmpeg, task[0], task[1], task[2], task[3], task[4]): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            index, saved_file, success, error_message = future.result()
            if success:
                stats["exported"] += 1
                print(f"✅ Highlight #{index} saved: {os.path.basename(saved_file)}")
            else:
                stats["failed_exports"] += 1
                print(f"❌ {error_message}")

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        raise SystemExit(
            "export_highlights now requires an AppConfig instance and should be called from main.py"
        )
