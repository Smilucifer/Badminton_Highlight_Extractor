# Boundary Refinement V2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the signal-first boundary refinement system truthful and measurable by adding real audio-assisted validation, a real confidence gate, truthful final statuses, and complete fallback metrics.

**Architecture:** Keep the current V2 heuristic pipeline intact, but strengthen `boundary_refinement.py` so motion-proposed anchors are validated with audio support and a score gate before acceptance. Keep next-serve protection based on `raw_start_time`, make final status reflect the final output rather than intermediate steps, and expose fallback metrics that truthfully measure dependency on default behavior.

**Tech Stack:** Python 3.11, dataclasses config layer, NumPy, librosa-derived energy points, existing vision velocity metrics, FFmpeg export pipeline, argparse

---

## File Map

- Modify: `boundary_refinement.py`
  - Add audio-aware candidate validation, signal gating, truthful status resolution, and fallback accounting helpers.
- Modify: `config.py`
  - Reuse existing V2 config, no new section required unless a minimal score-weight knob is needed.
- Modify: `main.py`
  - Add explicit fallback-rate and unchanged-count reporting.
- Modify: `README.md`
  - Align the docs with what the implementation truly does in V2.1.
- Modify: `docs/superpowers/specs/2026-04-17-boundary-refinement-v2.1-design.md`
  - Add validation notes after implementation details are concrete.

### Task 1: Add Audio-Assisted Candidate Validation Helpers

**Files:**
- Modify: `boundary_refinement.py`

- [ ] **Step 1: Write the failing helper expectation**

```python
from boundary_refinement import _score_start_candidate, _score_end_candidate
```

- [ ] **Step 2: Run the import to verify it fails**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from boundary_refinement import _score_start_candidate, _score_end_candidate"`
Expected: FAIL with `ImportError` because the scoring helpers do not exist yet

- [ ] **Step 3: Add an audio-window helper for local energy support**

```python
def _energy_change(
    energy_points: list[tuple[float, float]],
    center_time: float,
    window: float,
) -> float:
    before = _window_average(energy_points, center_time - window, window)
    after = _window_average(energy_points, center_time + window, window)
    return after - before
```

- [ ] **Step 4: Add start-candidate and end-candidate scoring helpers**

```python
def _score_start_candidate(
    candidate_time: float,
    raw_start: float,
    velocity_points: list[tuple[float, float]],
    energy_points: list[tuple[float, float]],
    valid_hits: list[float],
    config: AppConfig,
) -> float:
    motion_delta = _window_average(velocity_points, candidate_time + config.boundary.state_window, config.boundary.state_window) - _window_average(velocity_points, candidate_time - config.boundary.state_window, config.boundary.state_window)
    audio_delta = _energy_change(energy_points, candidate_time, config.boundary.state_window)
    first_hit = valid_hits[0] if valid_hits else raw_start
    constraint_bonus = 1.0 if candidate_time <= first_hit and first_hit - candidate_time <= config.boundary.start_search_window else 0.0
    motion_score = 1.0 if motion_delta >= config.boundary.motion_rise_delta else 0.0
    audio_score = 1.0 if audio_delta > 0 else 0.0
    return motion_score + audio_score + constraint_bonus


def _score_end_candidate(
    candidate_time: float,
    raw_end: float,
    velocity_points: list[tuple[float, float]],
    energy_points: list[tuple[float, float]],
    valid_hits: list[float],
    config: AppConfig,
) -> float:
    motion_delta = _window_average(velocity_points, candidate_time - config.boundary.state_window, config.boundary.state_window) - _window_average(velocity_points, candidate_time + config.boundary.state_window, config.boundary.state_window)
    audio_delta = -_energy_change(energy_points, candidate_time, config.boundary.state_window)
    last_hit = valid_hits[-1] if valid_hits else raw_end
    constraint_bonus = 1.0 if candidate_time >= last_hit and candidate_time - last_hit <= config.boundary.end_search_window else 0.0
    motion_score = 1.0 if motion_delta >= config.boundary.motion_fall_delta else 0.0
    audio_score = 1.0 if audio_delta > 0 else 0.0
    return motion_score + audio_score + constraint_bonus
```

- [ ] **Step 5: Run syntax verification after adding the scoring helpers**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile boundary_refinement.py config.py`
Expected: PASS with no output

- [ ] **Step 6: Commit the scoring helpers**

```bash
git add boundary_refinement.py config.py
git commit -m "feat: add audio-assisted refinement scoring helpers"
```

### Task 2: Make `signal_match_min_score` Gate Start and End Acceptance

**Files:**
- Modify: `boundary_refinement.py`

- [ ] **Step 1: Write the failing score-gate expectation**

```python
candidate_score = 0.5
signal_match_min_score = 1.0
assert candidate_score >= signal_match_min_score
```

- [ ] **Step 2: Run the simple expectation check to confirm the example fails**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "candidate_score = 0.5; signal_match_min_score = 1.0; print(candidate_score >= signal_match_min_score)"`
Expected: PASS with output `False`

- [ ] **Step 3: Update `_find_start_anchor()` to score candidates before accepting them**

```python
def _find_start_anchor(...):
    ...
    if rise_anchor is not None:
        score = _score_start_candidate(
            rise_anchor,
            raw_start,
            velocity_points,
            energy_points,
            valid_hits,
            config,
        )
        if score >= config.boundary.signal_match_min_score:
            return rise_anchor, "signal_matched_motion_rise"
```

- [ ] **Step 4: Update `_find_end_anchor()` to score candidates before accepting them**

```python
def _find_end_anchor(...):
    ...
    if cooldown_anchor is not None:
        score = _score_end_candidate(
            cooldown_anchor,
            raw_end,
            velocity_points,
            energy_points,
            valid_hits,
            config,
        )
        if score >= config.boundary.signal_match_min_score:
            return cooldown_anchor, "signal_matched_motion_cooldown"
```

- [ ] **Step 5: Thread `energy_points` and rally `valid_hits` through the refinement helper calls**

```python
start_anchor, start_reason = _find_start_anchor(
    raw_start,
    velocity_points,
    energy_points,
    rally.get("valid_hits", []),
    config,
)

end_anchor, end_reason = _find_end_anchor(
    raw_end,
    velocity_points,
    energy_points,
    rally.get("valid_hits", []),
    config,
)
```

- [ ] **Step 6: Run syntax verification after the score-gate changes**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile boundary_refinement.py config.py`
Expected: PASS with no output

- [ ] **Step 7: Commit the real signal gate**

```bash
git add boundary_refinement.py config.py
git commit -m "feat: gate refinement anchors by signal confidence"
```

### Task 3: Make Final Status Truthful About Next-Serve Clamping

**Files:**
- Modify: `boundary_refinement.py`

- [ ] **Step 1: Write the failing final-status expectation**

```python
status = "signal_matched_end_only"
end_reason = "next_serve_protection"
assert status == "clamped_by_next_serve"
```

- [ ] **Step 2: Run the simple expectation check to confirm the example fails**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "status = 'signal_matched_end_only'; end_reason = 'next_serve_protection'; print(status == 'clamped_by_next_serve')"`
Expected: PASS with output `False`

- [ ] **Step 3: Update status construction to reflect final output ownership**

```python
def _build_refinement_status(start_reason: str, end_reason: str) -> str:
    if end_reason == "next_serve_protection":
        return "clamped_by_next_serve"
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

- [ ] **Step 4: Keep per-side reason fields unchanged, but let the rally-level status reflect the final override**

```python
result["boundary_reason_start"] = start_reason
result["boundary_reason_end"] = end_reason
result["refinement_status"] = _build_refinement_status(start_reason, end_reason)
```

- [ ] **Step 5: Run syntax verification after the truthful-status change**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile boundary_refinement.py config.py`
Expected: PASS with no output

- [ ] **Step 6: Commit the status fix**

```bash
git add boundary_refinement.py config.py
git commit -m "feat: make refinement status reflect final output"
```

### Task 4: Add Complete Fallback and Unchanged Metrics

**Files:**
- Modify: `main.py`
- Modify: `boundary_refinement.py`

- [ ] **Step 1: Write the failing metric expectation**

```python
metrics = {}
assert "start_fallback_rate" in metrics
assert "end_fallback_rate" in metrics
assert "full_fallback_rate" in metrics
assert "unchanged_count" in metrics
```

- [ ] **Step 2: Run the simple structural check to confirm the fields do not exist yet**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "metrics = {}; print('start_fallback_rate' in metrics, 'unchanged_count' in metrics)"`
Expected: PASS with output `False False`

- [ ] **Step 3: Add helper predicates in `boundary_refinement.py` if needed for fallback detection**

```python
def _is_start_fallback(reason: str) -> bool:
    return reason == "fallback_front_pad"


def _is_end_fallback(reason: str) -> bool:
    return reason == "fallback_back_pad"
```

- [ ] **Step 4: Compute `unchanged_count` and explicit fallback rates in `main.py`**

```python
start_fallback_count = sum(1 for rally in refined_rallies if rally.get("boundary_reason_start") == "fallback_front_pad")
end_fallback_count = sum(1 for rally in refined_rallies if rally.get("boundary_reason_end") == "fallback_back_pad")
full_fallback_count = sum(1 for rally in refined_rallies if rally.get("refinement_status") == "fallback_both")
unchanged_count = sum(
    1
    for rally in refined_rallies
    if rally.get("raw_start_time") == rally.get("start_time") and rally.get("raw_end_time") == rally.get("end_time")
)

total_refined = max(1, len(refined_rallies))
start_fallback_rate = round(start_fallback_count / total_refined, 3)
end_fallback_rate = round(end_fallback_count / total_refined, 3)
full_fallback_rate = round(full_fallback_count / total_refined, 3)
```

- [ ] **Step 5: Print the new metrics in the per-video refinement summary**

```python
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
```

- [ ] **Step 6: Run syntax verification after the metric expansion**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -m py_compile boundary_refinement.py main.py config.py`
Expected: PASS with no output

- [ ] **Step 7: Commit the truthful metrics**

```bash
git add boundary_refinement.py main.py config.py
git commit -m "feat: report truthful fallback and unchanged metrics"
```

### Task 5: Align Docs With Actual V2.1 Behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-17-boundary-refinement-v2.1-design.md`

- [ ] **Step 1: Write the failing documentation expectation**

The README should explicitly say that audio is a supporting validation signal and that fallback rates are reported to measure signal-first trustworthiness.

- [ ] **Step 2: Confirm the current README does not yet contain the final V2.1 wording**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8').lower(); print('supporting validation signal' in text, 'trustworthiness' in text)"`
Expected: PASS with output `False False`

- [ ] **Step 3: Update the README with a truthful V2.1 explanation**

```markdown
### Boundary refinement trust signals / 边界精修可信信号
Version 2.1 treats motion as the primary anchor signal and audio as a supporting validation signal. A boundary is only accepted as signal-matched when the local evidence is strong enough; otherwise the system falls back to conservative default behavior.

Important metrics:
- `signal_refined_count`
- `start_fallback_rate`
- `end_fallback_rate`
- `full_fallback_rate`
- `unchanged_count`
```

- [ ] **Step 4: Add explicit trust-validation notes to the V2.1 spec**

```markdown
## Implementation validation notes
- Verify that audio affects acceptance, not just metadata flow
- Verify that `signal_match_min_score` changes acceptance behavior
- Verify that next-serve clamping changes `refinement_status` to `clamped_by_next_serve`
- Verify that fallback and unchanged metrics are populated per video
```

- [ ] **Step 5: Verify the README contains the expected V2.1 wording**

Run: `"D:\ClaudeWorkspace\Code\badminton_video_autocut\.venv\Scripts\python.exe" -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8').lower(); print('supporting validation signal' in text, 'trustworthiness' in text)"`
Expected: PASS with output `True True`

- [ ] **Step 6: Commit the V2.1 documentation updates**

```bash
git add README.md docs/superpowers/specs/2026-04-17-boundary-refinement-v2.1-design.md
git commit -m "docs: explain v2.1 refinement trust metrics"
```

## Self-Review

- Spec coverage check: This plan covers audio-assisted validation, confidence gating, truthful final status, complete fallback metrics, unchanged metrics, and documentation alignment. No approved V2.1 design requirement is left without a task.
- Placeholder scan: No `TODO`, `TBD`, or vague “handle appropriately” instructions remain; each code-changing step contains explicit code or exact commands.
- Type consistency: `signal_match_min_score` and the V2 boundary fields already exist before the scoring and gating tasks use them, `refinement_status` is updated before metrics consume it, and the V2.1 docs are aligned only after the implementation semantics are made truthful.
