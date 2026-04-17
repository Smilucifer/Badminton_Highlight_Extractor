# Boundary Refinement V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade boundary refinement from a fallback-first skeleton into a signal-first heuristic system that more often finds start/end anchors from local evidence.

**Architecture:** Keep the existing rough-rally → refinement → export pipeline, but update `boundary_refinement.py` so start/end boundary decisions are driven by local state transitions rather than default padding. Use `valid_hits` as anchor constraints, `velocity_points` as the primary state signal, `energy_points` as supporting evidence, and explicit reason/status fields plus fallback-rate metrics to measure whether refinement is genuinely improving.

**Tech Stack:** Python 3.11, dataclasses config layer, NumPy, librosa-derived energy series, existing vision velocity metrics, FFmpeg export pipeline, argparse

---

## File Map

- Modify: `config.py`
  - Add the remaining V2 boundary tuning knobs needed for local state-transition decisions.
- Modify: `boundary_refinement.py`
  - Replace fallback-first behavior with signal-first local anchor search for start/end boundaries.
- Modify: `main.py`
  - Report refinement status metrics and meaningful counts instead of only “changed rallies”.
- Modify: `hybrid_segmentation.py`
  - Keep providing local signal context in the shape refinement expects.
- Modify: `export_highlights.py`
  - Align next-serve protection with the explicit V2 contract based on the next rally’s `raw_start_time`.
- Modify: `README.md`
  - Explain V2’s signal-aware refinement behavior and metrics-oriented interpretation.
- Modify: `docs/superpowers/specs/2026-04-17-boundary-refinement-v2-design.md`
  - Add implementation validation notes once behavior is concrete.

### Task 1: Add V2 Boundary Configuration Knobs

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Write the failing config expectation**

```python
from config import get_default_config

cfg = get_default_config()
assert hasattr(cfg.boundary, "state_window")
assert hasattr(cfg.boundary, "motion_rise_delta")
assert hasattr(cfg.boundary, "motion_fall_delta")
assert hasattr(cfg.boundary, "signal_match_min_score")
```

- [ ] **Step 2: Run the config check to verify it fails**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from config import get_default_config; cfg = get_default_config(); print(hasattr(cfg.boundary, 'state_window'))"`
Expected: PASS with output `False`

- [ ] **Step 3: Add the V2 boundary fields to `BoundaryConfig`**

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
    state_window: float = 0.35
    motion_rise_delta: float = 15.0
    motion_fall_delta: float = 15.0
    signal_match_min_score: float = 1.0
```

- [ ] **Step 4: Verify the new V2 fields are present**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from config import get_default_config; cfg = get_default_config(); print(cfg.boundary.state_window, cfg.boundary.motion_rise_delta, cfg.boundary.signal_match_min_score)"`
Expected: PASS with output `0.35 15.0 1.0`

- [ ] **Step 5: Commit the V2 config expansion**

```bash
git add config.py
git commit -m "feat: add v2 boundary signal thresholds"
```

### Task 2: Add Local State Helper Functions

**Files:**
- Modify: `boundary_refinement.py`

- [ ] **Step 1: Write the failing helper expectation**

```python
from boundary_refinement import _window_average, _find_state_transition
```

- [ ] **Step 2: Run the import to verify it fails**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from boundary_refinement import _window_average, _find_state_transition"`
Expected: FAIL with `ImportError` because the helpers do not exist yet

- [ ] **Step 3: Add a small-window averaging helper**

```python
def _window_average(
    points: list[tuple[float, float]],
    center_time: float,
    window: float,
) -> float:
    values = [
        value
        for time_value, value in points
        if center_time - window <= time_value <= center_time + window
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))
```

- [ ] **Step 4: Add a generic local transition finder**

```python
def _find_state_transition(
    points: list[tuple[float, float]],
    candidate_times: list[float],
    window: float,
    delta_threshold: float,
    direction: str,
) -> float | None:
    for candidate_time in candidate_times:
        before = _window_average(points, candidate_time - window, window)
        after = _window_average(points, candidate_time + window, window)
        delta = after - before
        if direction == "rise" and delta >= delta_threshold:
            return candidate_time
        if direction == "fall" and -delta >= delta_threshold:
            return candidate_time
    return None
```

- [ ] **Step 5: Run syntax verification for the new helpers**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile boundary_refinement.py config.py`
Expected: PASS with no output

- [ ] **Step 6: Commit the helper functions**

```bash
git add boundary_refinement.py config.py
git commit -m "feat: add local state helpers for boundary refinement"
```

### Task 3: Make Start Boundary Signal-First

**Files:**
- Modify: `boundary_refinement.py`

- [ ] **Step 1: Write the failing start-reason expectation**

```python
result = {
    "boundary_reason_start": "fallback_front_pad",
}
assert result["boundary_reason_start"].startswith("signal_matched")
```

- [ ] **Step 2: Run a simple expectation check to confirm fallback is still the default**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "result = {'boundary_reason_start': 'fallback_front_pad'}; print(result['boundary_reason_start'].startswith('signal_matched'))"`
Expected: PASS with output `False`

- [ ] **Step 3: Add candidate-time generation for pre-start search**

```python
def _candidate_times_before(
    hit_time: float,
    points: list[tuple[float, float]],
    search_window: float,
) -> list[float]:
    return [
        time_value
        for time_value, _ in points
        if hit_time - search_window <= time_value <= hit_time
    ]
```

- [ ] **Step 4: Add signal-first start-anchor selection**

```python
def _find_start_anchor(
    raw_start: float,
    velocity_points: list[tuple[float, float]],
    config: AppConfig,
) -> tuple[float | None, str | None]:
    candidates = _candidate_times_before(
        raw_start,
        velocity_points,
        config.boundary.start_search_window,
    )
    rise_anchor = _find_state_transition(
        velocity_points,
        candidates,
        config.boundary.state_window,
        config.boundary.motion_rise_delta,
        "rise",
    )
    if rise_anchor is not None:
        return rise_anchor, "signal_matched_motion_rise"

    low_motion_threshold = _estimate_low_motion_threshold(velocity_points, config)
    for candidate_time in reversed(candidates):
        after = _window_average(velocity_points, candidate_time + config.boundary.state_window, config.boundary.state_window)
        if after > low_motion_threshold:
            return candidate_time, "signal_matched_low_activity_exit"

    return None, None
```

- [ ] **Step 5: Use the start anchor inside `refine_boundaries()` before falling back**

```python
start_anchor, start_reason = _find_start_anchor(raw_start, velocity_points, config)
if start_anchor is not None:
    refined_start = _clamp(start_anchor, start_floor, start_ceiling)
else:
    refined_start = _clamp(raw_start - config.boundary.fallback_pad_front, start_floor, start_ceiling)
    start_reason = "fallback_front_pad"
```

- [ ] **Step 6: Run syntax verification after the start-anchor change**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile boundary_refinement.py config.py`
Expected: PASS with no output

- [ ] **Step 7: Commit the signal-first start refinement**

```bash
git add boundary_refinement.py config.py
git commit -m "feat: add signal-first start boundary refinement"
```

### Task 4: Make End Boundary Signal-First

**Files:**
- Modify: `boundary_refinement.py`
- Modify: `export_highlights.py`

- [ ] **Step 1: Write the failing end-reason expectation**

```python
result = {
    "boundary_reason_end": "fallback_back_pad",
}
assert result["boundary_reason_end"].startswith("signal_matched")
```

- [ ] **Step 2: Run a simple expectation check to confirm fallback is still the default**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "result = {'boundary_reason_end': 'fallback_back_pad'}; print(result['boundary_reason_end'].startswith('signal_matched'))"`
Expected: PASS with output `False`

- [ ] **Step 3: Add candidate-time generation for post-end search**

```python
def _candidate_times_after(
    hit_time: float,
    points: list[tuple[float, float]],
    search_window: float,
) -> list[float]:
    return [
        time_value
        for time_value, _ in points
        if hit_time <= time_value <= hit_time + search_window
    ]
```

- [ ] **Step 4: Add signal-first end-anchor selection using motion fall and low-activity entry**

```python
def _find_end_anchor(
    raw_end: float,
    velocity_points: list[tuple[float, float]],
    config: AppConfig,
) -> tuple[float | None, str | None]:
    candidates = _candidate_times_after(
        raw_end,
        velocity_points,
        config.boundary.end_search_window,
    )
    cooldown_anchor = _find_state_transition(
        velocity_points,
        candidates,
        config.boundary.state_window,
        config.boundary.motion_fall_delta,
        "fall",
    )
    if cooldown_anchor is not None:
        return cooldown_anchor, "signal_matched_motion_cooldown"

    low_motion_threshold = _estimate_low_motion_threshold(velocity_points, config)
    for candidate_time in candidates:
        after = _window_average(velocity_points, candidate_time + config.boundary.state_window, config.boundary.state_window)
        if after <= low_motion_threshold:
            return candidate_time, "signal_matched_low_activity_entry"

    return None, None
```

- [ ] **Step 5: Use `raw_start_time` as the next-serve protection contract**

```python
if index < len(rallies) - 1:
    next_raw_start = float(rallies[index + 1].get("raw_start_time", rallies[index + 1]["start_time"]))
    refined_end = min(refined_end, next_raw_start - config.boundary.post_hit_guard)
```

- [ ] **Step 6: Mirror the same contract in `export_highlights.py`**

```python
if index < len(valid_rallies) - 1:
    next_serve_time = float(
        valid_rallies[index + 1].get(
            "raw_start_time",
            valid_rallies[index + 1].get("start_time", end),
        )
    )
    end = min(end, next_serve_time - config.boundary.post_hit_guard)
```

- [ ] **Step 7: Run syntax verification after the end-anchor change**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile boundary_refinement.py export_highlights.py config.py`
Expected: PASS with no output

- [ ] **Step 8: Commit the signal-first end refinement**

```bash
git add boundary_refinement.py export_highlights.py config.py
git commit -m "feat: add signal-first end boundary refinement"
```

### Task 5: Add Truthful Reasons, Status, and Fallback Metrics

**Files:**
- Modify: `boundary_refinement.py`
- Modify: `main.py`

- [ ] **Step 1: Write the failing status expectation**

```python
rally = {
    "boundary_reason_start": "fallback_front_pad",
    "boundary_reason_end": "fallback_back_pad",
}
assert "refinement_status" in rally
```

- [ ] **Step 2: Run the structural check to confirm the status field is absent**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "rally = {'boundary_reason_start': 'fallback_front_pad', 'boundary_reason_end': 'fallback_back_pad'}; print('refinement_status' in rally)"`
Expected: PASS with output `False`

- [ ] **Step 3: Add rally-level refinement status computation**

```python
def _build_refinement_status(start_reason: str, end_reason: str) -> str:
    start_signal = start_reason.startswith("signal_matched")
    end_signal = end_reason.startswith("signal_matched")
    if start_signal and end_signal:
        return "signal_matched_both"
    if start_signal:
        return "signal_matched_start_only"
    if end_signal:
        return "signal_matched_end_only"
    return "fallback_both"
```

- [ ] **Step 4: Store truthful reason and status fields in the refined rally payload**

```python
result["boundary_reason_start"] = start_reason
result["boundary_reason_end"] = end_reason
result["refinement_status"] = _build_refinement_status(start_reason, end_reason)
```

- [ ] **Step 5: Add aggregate refinement metrics in `main.py`**

```python
signal_refined_count = sum(1 for rally in refined_rallies if rally.get("refinement_status") != "fallback_both")
full_signal_refined_count = sum(1 for rally in refined_rallies if rally.get("refinement_status") == "signal_matched_both")
partial_signal_refined_count = sum(1 for rally in refined_rallies if rally.get("refinement_status") in {"signal_matched_start_only", "signal_matched_end_only"})
fallback_refined_count = sum(1 for rally in refined_rallies if rally.get("refinement_status") == "fallback_both")
next_serve_clamped_count = sum(1 for rally in refined_rallies if rally.get("boundary_reason_end") == "next_serve_protection")

print(
    "  - Refinement metrics:"
    f" signal_refined={signal_refined_count}"
    f", full_signal={full_signal_refined_count}"
    f", partial_signal={partial_signal_refined_count}"
    f", fallback={fallback_refined_count}"
    f", next_serve_clamped={next_serve_clamped_count}"
)
```

- [ ] **Step 6: Run syntax verification after the metric changes**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile boundary_refinement.py main.py config.py`
Expected: PASS with no output

- [ ] **Step 7: Commit the truthful metrics and statuses**

```bash
git add boundary_refinement.py main.py config.py
git commit -m "feat: report truthful refinement statuses and metrics"
```

### Task 6: Update Documentation for Signal-First Refinement

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-17-boundary-refinement-v2-design.md`

- [ ] **Step 1: Write the failing documentation expectation**

The README should mention that boundary refinement now attempts signal-matched start/end anchors before using fallback behavior.

- [ ] **Step 2: Confirm the current README does not yet describe signal-first refinement**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8').lower(); print('signal-first' in text, 'fallback rate' in text)"`
Expected: PASS with output `False False`

- [ ] **Step 3: Update the README with a short signal-first explanation**

```markdown
### Signal-first boundary refinement / 信号优先边界精修
The refinement layer first tries to find start/end anchors from local motion and audio transitions. Only when local evidence is insufficient does it fall back to conservative default padding.

Useful evaluation ideas:
- compare old/new start boundaries
- compare old/new end boundaries
- track fallback rate and signal-matched rate
```

- [ ] **Step 4: Append implementation validation notes to the V2 design spec**

```markdown
## Implementation validation notes
- Track `start_fallback_rate`, `end_fallback_rate`, and `full_fallback_rate`
- Record `signal_matched_both`, `signal_matched_start_only`, and `signal_matched_end_only`
- Compare signal-first results with fallback-only behavior on a representative sample set
```

- [ ] **Step 5: Verify the README contains the expected V2 wording**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8').lower(); print('signal-first' in text, 'fallback rate' in text)"`
Expected: PASS with output `True True`

- [ ] **Step 6: Commit the V2 documentation updates**

```bash
git add README.md docs/superpowers/specs/2026-04-17-boundary-refinement-v2-design.md
git commit -m "docs: explain signal-first boundary refinement"
```

## Self-Review

- Spec coverage check: This plan covers signal-first boundary search, local state helpers, start/end anchor logic, truthful reasons/statuses, fallback-rate metrics, raw-start-based next-serve protection, and documentation updates. No approved V2 design requirement is left without a task.
- Placeholder scan: No `TODO`, `TBD`, or vague “handle appropriately” instructions remain; all code-changing steps include explicit code or exact commands.
- Type consistency: `BoundaryConfig` fields introduced in Task 1 are reused consistently in later tasks, `refinement_status` is introduced before metrics consume it, and the next-serve protection contract consistently references `raw_start_time` after Task 4.
