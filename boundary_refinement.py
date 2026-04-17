from __future__ import annotations

from typing import Any

import numpy as np

from config import AppConfig


def _estimate_low_motion_threshold(
    velocity_points: list[tuple[float, float]],
    config: AppConfig,
) -> float:
    values = [value for _, value in velocity_points]
    if not values:
        return 0.0
    return float(np.percentile(values, config.boundary.low_motion_percentile))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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


def _energy_change(
    energy_points: list[tuple[float, float]],
    center_time: float,
    window: float,
) -> float:
    before = _window_average(energy_points, center_time - window, window)
    after = _window_average(energy_points, center_time + window, window)
    return after - before


def _score_start_candidate(
    candidate_time: float,
    raw_start: float,
    velocity_points: list[tuple[float, float]],
    energy_points: list[tuple[float, float]],
    valid_hits: list[float],
    config: AppConfig,
) -> float:
    motion_delta = _window_average(
        velocity_points,
        candidate_time + config.boundary.state_window,
        config.boundary.state_window,
    ) - _window_average(
        velocity_points,
        candidate_time - config.boundary.state_window,
        config.boundary.state_window,
    )
    audio_delta = _energy_change(energy_points, candidate_time, config.boundary.state_window)
    first_hit = valid_hits[0] if valid_hits else raw_start
    constraint_bonus = (
        1.0
        if candidate_time <= first_hit
        and first_hit - candidate_time <= config.boundary.start_search_window
        else 0.0
    )
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
    motion_delta = _window_average(
        velocity_points,
        candidate_time - config.boundary.state_window,
        config.boundary.state_window,
    ) - _window_average(
        velocity_points,
        candidate_time + config.boundary.state_window,
        config.boundary.state_window,
    )
    audio_delta = -_energy_change(energy_points, candidate_time, config.boundary.state_window)
    last_hit = valid_hits[-1] if valid_hits else raw_end
    constraint_bonus = (
        1.0
        if candidate_time >= last_hit
        and candidate_time - last_hit <= config.boundary.end_search_window
        else 0.0
    )
    motion_score = 1.0 if motion_delta >= config.boundary.motion_fall_delta else 0.0
    audio_score = 1.0 if audio_delta > 0 else 0.0
    return motion_score + audio_score + constraint_bonus


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


def _find_start_anchor(
    raw_start: float,
    velocity_points: list[tuple[float, float]],
    energy_points: list[tuple[float, float]],
    valid_hits: list[float],
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

    low_motion_threshold = _estimate_low_motion_threshold(velocity_points, config)
    for candidate_time in reversed(candidates):
        after = _window_average(
            velocity_points,
            candidate_time + config.boundary.state_window,
            config.boundary.state_window,
        )
        if after > low_motion_threshold:
            score = _score_start_candidate(
                candidate_time,
                raw_start,
                velocity_points,
                energy_points,
                valid_hits,
                config,
            )
            if score >= config.boundary.signal_match_min_score:
                return candidate_time, "signal_matched_low_activity_exit"

    return None, None


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


def _find_end_anchor(
    raw_end: float,
    velocity_points: list[tuple[float, float]],
    energy_points: list[tuple[float, float]],
    valid_hits: list[float],
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

    low_motion_threshold = _estimate_low_motion_threshold(velocity_points, config)
    for candidate_time in candidates:
        after = _window_average(
            velocity_points,
            candidate_time + config.boundary.state_window,
            config.boundary.state_window,
        )
        if after <= low_motion_threshold:
            score = _score_end_candidate(
                candidate_time,
                raw_end,
                velocity_points,
                energy_points,
                valid_hits,
                config,
            )
            if score >= config.boundary.signal_match_min_score:
                return candidate_time, "signal_matched_low_activity_entry"

    return None, None


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

        start_anchor, start_reason = _find_start_anchor(
            raw_start,
            velocity_points,
            energy_points,
            rally.get("valid_hits", []),
            config,
        )
        if start_anchor is not None:
            refined_start = _clamp(start_anchor, start_floor, start_ceiling)
        else:
            refined_start = _clamp(
                raw_start - config.boundary.fallback_pad_front,
                start_floor,
                start_ceiling,
            )
            start_reason = "fallback_front_pad"

        end_anchor, end_reason = _find_end_anchor(
            raw_end,
            velocity_points,
            energy_points,
            rally.get("valid_hits", []),
            config,
        )
        if end_anchor is not None:
            refined_end = _clamp(end_anchor, end_floor, end_ceiling)
        else:
            refined_end = _clamp(
                raw_end + config.boundary.fallback_pad_back,
                end_floor,
                end_ceiling,
            )
            end_reason = "fallback_back_pad"

        next_start = None
        if index < len(rallies) - 1:
            next_start = float(
                rallies[index + 1].get("raw_start_time", rallies[index + 1]["start_time"])
            )
            protected_end = next_start - config.boundary.post_hit_guard
            if refined_end > protected_end:
                refined_end = protected_end
                end_reason = "next_serve_protection"

        result = dict(rally)
        result["start_time"] = round(refined_start, 2)
        result["end_time"] = round(max(result["start_time"], refined_end), 2)
        result["duration"] = round(result["end_time"] - result["start_time"], 2)
        result["boundary_adjustment_front"] = round(result["start_time"] - raw_start, 2)
        result["boundary_adjustment_back"] = round(result["end_time"] - raw_end, 2)
        result["boundary_reason_start"] = start_reason
        result["boundary_reason_end"] = end_reason
        result["refinement_status"] = _build_refinement_status(start_reason, end_reason)
        result["boundary_low_motion_threshold"] = round(low_motion_threshold, 2)
        refined.append(result)

    return refined
