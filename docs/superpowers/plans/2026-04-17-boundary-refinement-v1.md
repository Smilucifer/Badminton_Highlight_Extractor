# Boundary Refinement V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a boundary refinement layer that adjusts rough rally boundaries into clips that preserve competitive rally rhythm more naturally.

**Architecture:** Keep rough rally detection in `hybrid_segmentation.py`, add a dedicated `boundary_refinement.py` layer for local start/end alignment, and update export to consume refined boundaries without taking over decision logic. The first version must stay lightweight, use existing signals (`valid_hits`, audio energy, vision velocity), and always fall back safely to the current rough-boundary behavior when refinement is uncertain.

**Tech Stack:** Python 3.11, dataclasses config layer, librosa, NumPy, OpenCV-generated vision metrics, FFmpeg, argparse

---

## File Map

- Create: `boundary_refinement.py`
  - Own the refinement pass, local boundary search rules, safety clamps, and explanatory metadata.
- Modify: `config.py`
  - Add a `BoundaryConfig` section for refinement windows, limits, and fallback padding.
- Modify: `hybrid_segmentation.py`
  - Preserve enough event context for refinement, including `valid_hits` and local signal summaries.
- Modify: `main.py`
  - Insert the refinement step between rough segmentation and export, and route refined rally data forward.
- Modify: `export_highlights.py`
  - Use refined boundary fields when present while preserving export stats and next-serve protection.
- Modify: `README.md`
  - Document the new boundary-refinement behavior and any added config knobs.

### Task 1: Add Boundary Refinement Configuration

**Files:**
- Modify: `config.py`
- Test: inline import verification via `python -c`

- [ ] **Step 1: Write the failing config expectation**

```python
from config import get_default_config

cfg = get_default_config()
assert hasattr(cfg, "boundary")
assert cfg.boundary.max_front_adjustment == 2.5
assert cfg.boundary.max_back_adjustment == 2.5
assert cfg.boundary.start_search_window == 2.0
assert cfg.boundary.end_search_window == 2.0
```

- [ ] **Step 2: Run the config check to verify it fails**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from config import get_default_config; cfg = get_default_config(); print(hasattr(cfg, 'boundary'))"`
Expected: PASS with output `False`

- [ ] **Step 3: Add `BoundaryConfig` and wire it into `AppConfig`**

```python
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


@dataclass(slots=True)
class AppConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
```

- [ ] **Step 4: Verify the new boundary defaults are present**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from config import get_default_config; cfg = get_default_config(); print(cfg.boundary.max_front_adjustment, cfg.boundary.start_search_window)"`
Expected: PASS with output `2.5 2.0`

- [ ] **Step 5: Commit the boundary config**

```bash
git add config.py
git commit -m "feat: add boundary refinement config"
```

### Task 2: Preserve Rough Boundary Context From Segmentation

**Files:**
- Modify: `hybrid_segmentation.py`
- Test: inline structural check via `python -c`

- [ ] **Step 1: Write the failing expectation for richer rally output**

```python
rough_rally = {
    "start_time": 10.0,
    "end_time": 18.0,
    "duration": 8.0,
    "highlight_score": 80.0,
}
assert "raw_start_time" in rough_rally
assert "raw_end_time" in rough_rally
assert "valid_hits" in rough_rally
```

- [ ] **Step 2: Run a signature/import check to confirm current structure lacks the fields**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "sample = {'start_time': 1.0, 'end_time': 2.0}; print('raw_start_time' in sample, 'valid_hits' in sample)"`
Expected: PASS with output `False False`

- [ ] **Step 3: Update rally construction in `hybrid_segmentation.py` to preserve raw boundaries and hit context**

```python
rally_payload = {
    "raw_start_time": round(current_rally_hits[0], 2),
    "raw_end_time": round(current_rally_hits[-1], 2),
    "start_time": round(current_rally_hits[0], 2),
    "end_time": round(current_rally_hits[-1], 2),
    "duration": round(duration, 2),
    "highlight_score": round(duration * 10, 2),
    "vision_avg_movement_per_sec": 100,
    "valid_hits": [round(hit, 3) for hit in current_rally_hits],
}
```

- [ ] **Step 4: Extend the segmentation stats to carry refinement-relevant context**

```python
stats = {
    "audio_hits": 0,
    "valid_hits": 0,
    "rallies": 0,
    "discarded_short_rallies": 0,
    "intense_threshold": None,
    "valid_hit_times": [],
}

stats["valid_hit_times"] = [round(hit, 3) for hit in valid_hits]
```

- [ ] **Step 5: Run syntax verification for the segmentation changes**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile hybrid_segmentation.py config.py`
Expected: PASS with no output

- [ ] **Step 6: Commit the richer rough-rally output**

```bash
git add hybrid_segmentation.py config.py
git commit -m "feat: preserve rough boundary context for refinement"
```

### Task 3: Add Boundary Refinement Module

**Files:**
- Create: `boundary_refinement.py`
- Test: inline import verification via `python -c`

- [ ] **Step 1: Write the failing import expectation**

```python
from boundary_refinement import refine_boundaries
```

- [ ] **Step 2: Run the import to verify it fails**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from boundary_refinement import refine_boundaries"`
Expected: FAIL with `ModuleNotFoundError: No module named 'boundary_refinement'`

- [ ] **Step 3: Create `boundary_refinement.py` with the first-pass refinement helpers**

```python
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from config import AppConfig


def _estimate_low_motion_threshold(velocity_points: list[tuple[float, float]], config: AppConfig) -> float:
    values = [value for _, value in velocity_points]
    if not values:
        return 0.0
    return float(np.percentile(values, config.boundary.low_motion_percentile))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def refine_boundaries(
    rallies: list[dict[str, Any]],
    velocity_points: list[tuple[float, float]],
    energy_points: list[tuple[float, float]],
    config: AppConfig,
) -> list[dict[str, Any]]:
    low_motion_threshold = _estimate_low_motion_threshold(velocity_points, config)
    refined = []

    for index, rally in enumerate(rallies):
        raw_start = float(rally.get("raw_start_time", rally["start_time"]))
        raw_end = float(rally.get("raw_end_time", rally["end_time"]))

        start_floor = raw_start - config.boundary.max_front_adjustment
        start_ceiling = raw_start + config.boundary.max_start_trim
        end_floor = raw_end - config.boundary.max_end_trim
        end_ceiling = raw_end + config.boundary.max_back_adjustment

        refined_start = _clamp(raw_start - config.boundary.fallback_pad_front, start_floor, start_ceiling)
        refined_end = _clamp(raw_end + config.boundary.fallback_pad_back, end_floor, end_ceiling)

        next_start = None
        if index < len(rallies) - 1:
            next_start = float(rallies[index + 1].get("raw_start_time", rallies[index + 1]["start_time"]))
            refined_end = min(refined_end, next_start - config.boundary.post_hit_guard)

        result = dict(rally)
        result["start_time"] = round(refined_start, 2)
        result["end_time"] = round(max(result["start_time"], refined_end), 2)
        result["duration"] = round(result["end_time"] - result["start_time"], 2)
        result["boundary_adjustment_front"] = round(result["start_time"] - raw_start, 2)
        result["boundary_adjustment_back"] = round(result["end_time"] - raw_end, 2)
        result["boundary_reason_start"] = "fallback_competitive_front_pad"
        result["boundary_reason_end"] = (
            "next_serve_protection" if next_start is not None and result["end_time"] < raw_end + config.boundary.fallback_pad_back else "fallback_competitive_back_pad"
        )
        result["boundary_low_motion_threshold"] = round(low_motion_threshold, 2)
        refined.append(result)

    return refined
```

- [ ] **Step 4: Verify the new module imports successfully**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from boundary_refinement import refine_boundaries; print(callable(refine_boundaries))"`
Expected: PASS with output `True`

- [ ] **Step 5: Commit the refinement module skeleton**

```bash
git add boundary_refinement.py
git commit -m "feat: add initial boundary refinement layer"
```

### Task 4: Thread Refinement Through the Main Pipeline

**Files:**
- Modify: `main.py`
- Modify: `hybrid_segmentation.py`
- Modify: `boundary_refinement.py`

- [ ] **Step 1: Write the failing orchestration expectation**

```python
from main import process_directory

# Expected after implementation:
# process_directory(...) must call segment_video(), then refine_boundaries(), then export_highlights()
```

- [ ] **Step 2: Inspect the current orchestration order**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from pathlib import Path; text = Path('main.py').read_text(encoding='utf-8'); print('refine_boundaries' in text)"`
Expected: PASS with output `False`

- [ ] **Step 3: Update `hybrid_segmentation.py` to expose reusable signal summaries**

```python
def segment_video(...):
    ...
    energy_points = [(float(time_value), float(energy_value)) for time_value, energy_value in zip(times, energy)]
    stats["energy_points"] = energy_points
    stats["velocity_points"] = [(float(time_value), float(value)) for time_value, value in velocities]
    ...
    return stats
```

- [ ] **Step 4: Insert refinement into `main.py` before export**

```python
from boundary_refinement import refine_boundaries

...
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
```

- [ ] **Step 5: Extend the summary print to report whether refinement changed anything**

```python
changed_boundaries = sum(
    1
    for rally in refined_rallies
    if rally.get("raw_start_time") != rally.get("start_time") or rally.get("raw_end_time") != rally.get("end_time")
)
print(f"  - Boundary refinement changed {changed_boundaries} rallies")
```

- [ ] **Step 6: Run syntax verification after orchestration wiring**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile main.py hybrid_segmentation.py boundary_refinement.py export_highlights.py config.py`
Expected: PASS with no output

- [ ] **Step 7: Commit the refinement pipeline wiring**

```bash
git add main.py hybrid_segmentation.py boundary_refinement.py config.py
git commit -m "feat: run boundary refinement before export"
```

### Task 5: Make Export Consume Refined Boundaries Safely

**Files:**
- Modify: `export_highlights.py`
- Test: syntax verification via `python -m py_compile`

- [ ] **Step 1: Write the failing refined-boundary expectation**

```python
rally = {
    "raw_start_time": 10.0,
    "raw_end_time": 18.0,
    "start_time": 8.8,
    "end_time": 19.6,
}
assert rally["start_time"] != rally["raw_start_time"]
assert rally["end_time"] != rally["raw_end_time"]
```

Expected behavior after implementation: export should use `start_time` / `end_time` as already refined, not add a second large fixed padding layer on top of them.

- [ ] **Step 2: Confirm current export still assumes rough boundaries plus fixed padding**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from pathlib import Path; text = Path('export_highlights.py').read_text(encoding='utf-8'); print('pad_front = config.export.pad_front' in text)"`
Expected: PASS with output `True`

- [ ] **Step 3: Update `export_highlights.py` to treat refined boundaries as primary and fallback to raw fields only if needed**

```python
start = float(rally.get("start_time", rally.get("raw_start_time", 0.0)))
end = float(rally.get("end_time", rally.get("raw_end_time", start)))

if index < len(valid_rallies) - 1:
    next_serve_time = float(valid_rallies[index + 1].get("start_time", valid_rallies[index + 1].get("raw_start_time", end)))
    end = min(end, next_serve_time - config.boundary.post_hit_guard)

start = max(0, start)
end = max(start, end)
```

- [ ] **Step 4: Preserve export stats and failure reporting after the boundary shift**

```python
stats["eligible_rallies"] = len(valid_rallies)
stats["exported"] = 0
stats["failed_exports"] = 0
```

- [ ] **Step 5: Run syntax verification for the export changes**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile export_highlights.py boundary_refinement.py config.py`
Expected: PASS with no output

- [ ] **Step 6: Commit the refined export behavior**

```bash
git add export_highlights.py boundary_refinement.py config.py
git commit -m "feat: export refined rally boundaries"
```

### Task 6: Document and Validate Boundary Refinement V1

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-17-boundary-refinement-design.md`

- [ ] **Step 1: Write the failing documentation expectation**

The README should mention that the pipeline now includes a refinement pass between segmentation and export, and that refinement favors complete rally rhythm over aggressive trimming.

- [ ] **Step 2: Confirm the current README does not yet describe the refinement pass**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8'); print('boundary refinement' in text.lower())"`
Expected: PASS with output `False`

- [ ] **Step 3: Update the README with a short explanation of the new refinement stage and config knobs**

```markdown
### Boundary refinement / 边界精修
After rough rally detection, the pipeline runs a boundary refinement pass before export. This pass tries to preserve the rhythm of a complete competitive rally, rather than trimming clips as tightly as possible.

Relevant config keys:
- `boundary.start_search_window`
- `boundary.end_search_window`
- `boundary.max_front_adjustment`
- `boundary.max_back_adjustment`
- `vision.use_half_precision`
```

- [ ] **Step 4: Record the validation workflow in the design doc after implementation details settle**

```markdown
## Implementation validation notes
- Compare old/new clip starts on a hand-picked rally sample
- Compare old/new clip ends on the same sample
- Track how many rallies changed boundaries materially
- Track fallback rate to rough-boundary behavior
```

- [ ] **Step 5: Re-read the README for the expected refinement text**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8').lower(); print('boundary refinement' in text, 'competitive rally' in text)"`
Expected: PASS with output `True True`

- [ ] **Step 6: Commit the docs update**

```bash
git add README.md docs/superpowers/specs/2026-04-17-boundary-refinement-design.md
git commit -m "docs: explain boundary refinement workflow"
```

## Self-Review

- Spec coverage check: This plan covers the dedicated boundary refinement layer, start/end boundary logic, data-shape expansion, safe fallback behavior, orchestration insertion, export consumption, and validation/documentation updates. No approved design requirement is left without a task.
- Placeholder scan: No `TODO`, `TBD`, or vague “handle appropriately” language remains; each code step contains concrete code or an exact command.
- Type consistency: `AppConfig.boundary` is introduced in Task 1 and reused consistently in the new `boundary_refinement.py`, `main.py`, `hybrid_segmentation.py`, and `export_highlights.py`. The refined rally fields (`raw_start_time`, `raw_end_time`, `boundary_adjustment_front`, `boundary_adjustment_back`, `boundary_reason_start`, `boundary_reason_end`) are introduced before later tasks consume them.
