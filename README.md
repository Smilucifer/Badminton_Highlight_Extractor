# Badminton Highlight Extractor 🏸

A hybrid audio-visual tool that automatically segments badminton match videos and clips the highest-value rallies into individual highlights.

## Features
- **Vision-Based Activity Tracking**: Uses YOLOv8-pose to track player movements on court and define baseline dynamic activity.
- **Audio Energy Detection**: Analyzes audio tracks to identify sharp peak sounds (hits/smashes) using `librosa`.
- **Hybrid Segmentation**: Fuses audio hit detection with visual movement tracking to filter out background noise (like walking or shoe squeaks) and perfectly identify real badminton rallies.
- **Smart Adaptive Clipping**: Accurately bounds boundaries for high-paced consecutive rallies, allowing overlapping highlights *without* accidentally catching the opponent's next serve.
- **Quality Filtering**: Automatically ignores simple rallies (like unforced errors on serve) and outputs only sustained multi-shot rallies (score threshold mechanism).

## Usage

**1. Extract basic stats**
```bash
python analyze_vision.py
python hybrid_segmentation.py
```

**2. Export Highlights**
```bash
python export_highlights.py
```

## Requirements
See `requirements.txt` for libraries. This project uses FFmpeg for stream-copying the cut videos efficiently.
