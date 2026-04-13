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

**1. Process bulk videos / 批量处理指定目录下的所有视频**
```bash
# Provide the directory containing your video files (.mp4, .mkv, .avi, .mov)
# 提供包含你要处理视频的绝对或相对目录
python main.py "/path/to/your/videos"
```

Enjoy your highlights grouped perfectly under `/path/to/your/videos/highlights/` ! 
所有切割好的集锦会自动输出到输入目录下的 `highlights` 文件夹中！ 

## Requirements / 运行环境
See `requirements.txt` for Python libraries. 
This project uses FFmpeg and heavily relies on CUDA for YOLO inference. 
需提前系统安装 FFmpeg（用于音视频无损流拷贝与抽音），并且强烈依赖 CUDA 以加速 YOLO 推理。
