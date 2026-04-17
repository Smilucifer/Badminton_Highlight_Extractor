from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class VisionConfig:
    model_path: str = "yolov8n-pose.pt"
    batch_size: int = 16
    num_workers: int = 4
    sample_fps: float = 5.0
    confidence_threshold: float = 0.5
    use_half_precision: bool = True


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int | None = None
    hop_length: int = 512
    peak_height: float = 0.015
    min_peak_distance_sec: float = 0.2


@dataclass(slots=True)
class FusionConfig:
    motion_percentile: float = 50.0
    hit_window_before: float = 0.2
    hit_window_after: float = 1.2
    min_intense_ratio: float = 0.05
    rally_gap_threshold: float = 2.5
    min_rally_duration: float = 1.0


@dataclass(slots=True)
class ExportConfig:
    min_highlight_score: float = 80.0
    pad_front: float = 1.5
    pad_back: float = 2.0
    max_workers: int = 5


@dataclass(slots=True)
class BoundaryConfig:
    start_search_window: float = 2.0
    end_search_window: float = 2.0
    max_front_adjustment: float = 2.5
    max_back_adjustment: float = 2.5
    max_start_trim: float = 0.5
    max_end_trim: float = 0.5
    low_motion_percentile: float = 25.0
    pre_hit_guard: float = 0.15
    post_hit_guard: float = 0.2
    fallback_pad_front: float = 1.5
    fallback_pad_back: float = 2.0
    state_window: float = 0.35
    motion_rise_delta: float = 15.0
    motion_fall_delta: float = 15.0
    signal_match_min_score: float = 2.0


@dataclass(slots=True)
class AppConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_default_config() -> AppConfig:
    return AppConfig()


def merge_config_overrides(config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    for section_name, values in overrides.items():
        if not hasattr(config, section_name) or not isinstance(values, dict):
            continue
        section = getattr(config, section_name)
        for key, value in values.items():
            if hasattr(section, key):
                setattr(section, key, value)
    return config
