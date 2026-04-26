# Badminton Highlight Extractor 🏸

A hybrid audio-visual tool that automatically segments badminton match videos and clips the highest-value rallies into individual highlights.

一款基于纯视觉和音频混合特征的羽毛球比赛自动剪辑工具，自动识别并将最精华的对打回合精准裁剪为视频高光集锦。

## Features / 功能特性
- **Vision-Based Activity Tracking (视觉运动追踪)**: Uses YOLOv8-pose to track player movements on court and define baseline dynamic activity. 使用 YOLOv8-pose 跟踪球员场上运动特征，界定活跃区间。
- **Audio Energy Detection (声能峰值检测)**: Analyzes audio tracks to identify sharp peak sounds (hits/smashes) using `librosa`. 使用 `librosa` 分析音频中的能量锐响（击球声与杀球声）。
- **Hybrid Segmentation (音视频混合切分)**: Fuses audio hit detection with visual movement tracking to filter out background noise (like walking or shoe squeaks) and perfectly identify real badminton rallies. 结合视觉活跃度过滤杂音（如鞋底摩擦声），精准定位真实回合。
- **Smart Adaptive Clipping (智能自适应裁剪)**: Accurately bounds boundaries for high-paced consecutive rallies, allowing overlapping highlights *without* accidentally catching the opponent's next serve. 对高强度的连续拉吊回合提供灵活的边界裁剪缓冲时间，允许必要的重叠但绝不会摄入下一回合的发球。
- **Quality Filtering (高质量过滤)**: Automatically ignores simple rallies (like unforced errors on serve) and outputs only sustained multi-shot rallies (score threshold mechanism). 自动忽略发球失误等低质量回合，仅输出具备多回合对抗的高分拉吊视频。

## Usage / 使用方法

You can now process videos dynamically through command-line arguments without modifying the scripts. 中间过程文件（音频、JSON）全部使用临时文件夹流转，不产生垃圾文件。

### Windows quick start / Windows 快速开始
1. Install FFmpeg and make sure `ffmpeg` works in a normal Windows terminal. 先安装 FFmpeg，并确认在 Windows 终端中直接输入 `ffmpeg` 可以运行。
2. Place `yolov8n-pose.pt` in the project root. 将 `yolov8n-pose.pt` 放在项目根目录下。
3. Install Python dependencies from `requirements.txt`. 根据 `requirements.txt` 安装 Python 依赖。
4. Run the project with a video directory path. 使用视频目录路径运行主程序。
5. If you want to override defaults, pass a JSON config file with nested sections like `vision`, `audio`, `fusion`, `export`, and `boundary`. 如果你想覆盖默认参数，可以传入一个 JSON 配置文件，按 `vision`、`audio`、`fusion`、`export`、`boundary` 分组。

```bash
python main.py "D:\path\to\your\videos" --config "D:\path\to\config.json"
```

Recommended workflow / 推荐工作流：
- Start from the bundled full config file `my_config.json`. 从项目自带的完整配置 `my_config.json` 开始改最省心。
- For a balanced Windows + CUDA setup, the current recommended values keep `vision.use_half_precision=true`, `vision.num_workers=4`, `fusion.rally_gap_threshold=2.4`, `export.min_highlight_score=80`, and `export.max_workers=4`. 对于 Windows + CUDA 的平衡模式，目前推荐保留 `vision.use_half_precision=true`、`vision.num_workers=4`、`fusion.rally_gap_threshold=2.4`、`export.min_highlight_score=80`、`export.max_workers=4`。
- See `CONFIG_GUIDE.zh-CN.md` for a full Chinese explanation of every parameter. 所有参数的中文说明见 `CONFIG_GUIDE.zh-CN.md`。

Example config / 配置示例：

```json
{
  "vision": {
    "model_path": "yolov8n-pose.pt",
    "batch_size": 16,
    "num_workers": 4,
    "sample_fps": 5.0,
    "confidence_threshold": 0.5,
    "use_half_precision": true
  },
  "audio": {
    "sample_rate": null,
    "hop_length": 512,
    "peak_height": 0.015,
    "min_peak_distance_sec": 0.2
  },
  "fusion": {
    "motion_percentile": 50.0,
    "hit_window_before": 0.2,
    "hit_window_after": 1.2,
    "min_intense_ratio": 0.05,
    "rally_gap_threshold": 2.4,
    "min_rally_duration": 1.0
  },
  "export": {
    "min_highlight_score": 80,
    "pad_front": 1.5,
    "pad_back": 2.0,
    "max_workers": 4
  },
  "boundary": {
    "start_search_window": 2.0,
    "end_search_window": 2.0,
    "max_front_adjustment": 2.5,
    "max_back_adjustment": 2.5,
    "max_start_trim": 0.5,
    "max_end_trim": 0.5,
    "low_motion_percentile": 25.0,
    "pre_hit_guard": 0.15,
    "post_hit_guard": 0.2,
    "fallback_pad_front": 1.5,
    "fallback_pad_back": 2.0,
    "state_window": 0.35,
    "motion_rise_delta": 15.0,
    "motion_fall_delta": 15.0,
    "signal_match_min_score": 2.0
  }
}
```

The tool scans the directory for supported videos (`.mp4`, `.mkv`, `.avi`, `.mov`) and writes clipped highlights to `D:\path\to\your\videos\highlights\`.
程序会扫描目录中的支持格式视频（`.mp4`、`.mkv`、`.avi`、`.mov`），并将裁剪后的高光输出到 `D:\path\to\your\videos\highlights\`。

### Boundary refinement trust signals / 边界精修可信信号
Version 2.1 treats motion as the primary anchor signal and audio as a supporting validation signal. A boundary is only accepted as signal-matched when the local evidence is strong enough; otherwise the system falls back to conservative default behavior. This keeps refinement metrics useful for trustworthiness instead of just showing arbitrary boundary movement.
在 v2.1 中，运动信号仍然是主要锚点信号，而音频信号负责辅助验证。只有局部证据足够强时，边界才会被视为 signal-matched；否则系统会回退到保守默认裁法。这样输出的 refinement 指标才能更可信，而不只是记录边界有没有随便移动。

Important metrics / 关键指标：
- `signal_refined_count`
- `start_fallback_rate`
- `end_fallback_rate`
- `full_fallback_rate`
- `unchanged_count`

## Requirements / 运行环境
See `requirements.txt` for Python libraries.
This project uses FFmpeg and heavily relies on CUDA for YOLO inference.
需提前系统安装 FFmpeg（用于音视频无损流拷贝与抽音），并且强烈依赖 CUDA 以加速 YOLO 推理。
