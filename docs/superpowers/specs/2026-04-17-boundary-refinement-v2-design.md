# Boundary Refinement V2 Signal-First Design

**Date:** 2026-04-17
**Scope:** Second design iteration for highlight boundary refinement, focused on turning the existing fallback-first refinement skeleton into a signal-first boundary decision layer.

## Goal

Reduce reliance on default fallback clipping by making boundary refinement genuinely use local signals to find start and end anchors. The system should more often determine boundaries from local rally evidence, instead of defaulting to fixed conservative expansion.

## Primary Success Criterion

Version 2 is successful if boundary refinement is **more often decided by local signals than by default fallback logic**.

This means the main metric of success is not “average clip changed” or “average clip got longer/shorter,” but:

- more clips using signal-matched boundary decisions
- lower fallback reliance on both sides
- clearer separation between “real refinement” and “safe fallback”

## Why Version 1 Is Not Enough

Version 1 established the right architecture:

- rough rally detection remains separate
- a dedicated `boundary_refinement.py` exists
- export consumes refined boundaries
- fallback behavior is safe and explicit

But Version 1 still behaves like this in practice:

- rough boundary exists
- fallback front/back expansion is applied
- next-rally clamp is applied
- metadata is attached

The missing piece is that local signals do not yet materially drive the decision.

That means Version 1 is a refinement framework, not a true signal-aware refinement system.

## Product Direction

Version 2 still keeps the same viewing bias established earlier:

- preserve complete competitive rally rhythm
- prefer signal-based alignment over aggressive trimming
- avoid clipping away meaningful start buildup or end completion

But compared with Version 1, the product focus shifts to this:

- **Version 1:** create a safe refinement layer
- **Version 2:** make the refinement layer actually decide boundaries from evidence

## High-Level Approach

Keep the current architecture intact, but change decision priority inside `refine_boundaries()`:

- **Version 1 priority:** fallback first, signal mostly unused
- **Version 2 priority:** signal first, fallback only when signal confidence is insufficient

The new decision model is:

1. anchor search window around rough boundary
2. local signal evaluation inside that window
3. signal-matched boundary selection if a valid anchor is found
4. safe fallback only when no anchor is reliable

## Boundary Signals

Version 2 continues to use lightweight, already-available signals:

- `valid_hits`
- `velocity_points`
- `energy_points`

### Signal roles

- `valid_hits` define where refinement is allowed to search around a rally
- `velocity_points` act as the primary signal for movement state
- `energy_points` act as a supporting signal for activity transition confidence

### Design principle

For Version 2:

- **vision is the primary state signal**
- **audio is the supporting confirmation signal**
- **valid hits are the anchor constraints**

This keeps the design cheap and explainable.

## Start Boundary Strategy

### Objective

Find the point before the first meaningful exchange where the clip most naturally transitions from pre-rally low activity into rally buildup.

### Start anchor definition

The preferred start anchor is:

- the last low-activity exit before the first rally hit cluster
- or the start of a sustained motion rise that leads into the first valid hits

### Search window

Search backward from `raw_start_time` using `boundary.start_search_window`.

The search must stay local. It must not expand freely into the prior rally or long dead-time setup.

### Signal-first rules

In the local pre-start window, the refinement logic should try these in order:

1. detect a **motion-rise anchor**
   - a point where short-window motion after the point is materially stronger than short-window motion before it
   - and where that rise leads toward the first valid hit cluster
2. detect a **low-activity exit anchor**
   - a point where a short low-activity period ends and the rally enters an active phase
3. if neither anchor is reliable, use fallback front padding

### Interpretation

The start boundary should represent:

- “this is where the point starts to become a rally”

not:

- “this is exactly the first hit”
- “this is the earliest possible frame we can include”

## End Boundary Strategy

### Objective

Find the point after the last meaningful exchange where the rally naturally exits active play and settles into post-point cooldown.

### End anchor definition

The preferred end anchor is:

- the first stable low-activity entry after the last valid hit cluster
- or the first clear motion cooldown point after rally intensity drops out

### Search window

Search forward from `raw_end_time` using `boundary.end_search_window`.

### Signal-first rules

In the local post-end window, the refinement logic should try these in order:

1. detect a **motion-cooldown anchor**
   - a point where short-window motion after the point is materially lower than short-window motion before it
   - and no new valid rally activity appears immediately afterward
2. detect a **low-activity entry anchor**
   - the first stable transition into a low-activity period after the rally ends
3. if neither anchor is reliable, use fallback back padding

### Interpretation

The end boundary should represent:

- “this is where the point clearly stops being active play”

not merely:

- “this is the last hit timestamp plus a generic tail”

## Local State Model

Version 2 should explicitly reason about four local states:

- low activity
- rising activity
- rally activity
- cooling activity

This does not require a separate model file or ML classifier. It can be implemented using short-window comparisons on existing signal sequences.

### Low activity

A local region counts as low activity when:

- motion stays below a low-activity threshold for a short continuous span
- and audio energy is not showing new strong rally-like continuation

### Rising activity

A region counts as rising when:

- short-window motion after a point is clearly higher than before it
- and the rise leads into the first valid-hit cluster

### Cooling activity

A region counts as cooling when:

- short-window motion after a point is clearly lower than before it
- and no new valid-hit continuation appears in the same local region

## Threshold and Transition Logic

Version 2 should keep the logic simple and explainable.

### Low-motion threshold

The existing `low_motion_percentile` should stop being metadata-only and become a real decision threshold.

The refinement pass should compute a per-video low-motion threshold from `velocity_points` and then use it to classify local windows as low-activity or non-low-activity.

### Transition detection

Instead of using raw single-point peaks, compare short windows around candidate points:

- for start anchors: compare “before” window vs “after” window for motion increase
- for end anchors: compare “before” window vs “after” window for motion decrease

This keeps the heuristic robust against local noise.

## Required Boundary Reasons

Version 2 must stop reporting fallback as if it were true signal alignment.

### Start reasons
- `signal_matched_motion_rise`
- `signal_matched_low_activity_exit`
- `fallback_front_pad`

### End reasons
- `signal_matched_motion_cooldown`
- `signal_matched_low_activity_entry`
- `next_serve_protection`
- `fallback_back_pad`

These reason values are part of the product contract for debugging and evaluation.

## Refinement Status

In addition to per-side reasons, Version 2 should add a rally-level status field.

Recommended values:

- `signal_matched_both`
- `signal_matched_start_only`
- `signal_matched_end_only`
- `fallback_both`
- `clamped_by_next_serve`

This gives an immediate view of whether refinement truly used signal evidence.

## Fallback Metrics

Version 2 must explicitly measure fallback dependence.

### Required metrics
- `start_fallback_rate`
- `end_fallback_rate`
- `full_fallback_rate`
- `signal_refined_count`
- `full_signal_refined_count`
- `partial_signal_refined_count`
- `fallback_refined_count`
- `next_serve_clamped_count`
- `unchanged_count`

### Why these metrics matter

They answer the right question:

- is the system learning to decide boundaries from evidence?

rather than the weaker question:

- did the timestamps move?

## Next-Serve Protection Contract

Version 2 must define this explicitly.

### Decision

Next-serve protection should be based on the **next rally’s `raw_start_time`**, not the next rally’s refined start time.

### Reasoning

- `raw_start_time` is the next rally’s core anchor
- refined next-start may be moved earlier for rhythm preservation
- protection should defend against cutting into the next rally’s core event, not its optional contextual expansion

This avoids refinement-on-refinement coupling.

## Data Volume Guidance

Version 2 should begin reducing the weight of refinement input context.

Current context passes full-length `energy_points` and `velocity_points`. That is acceptable for V1/V2 prototyping, but the intended direction is:

- keep full context available if needed
- prefer local window slicing inside refinement
- avoid building later features that assume the entire sequence is always required everywhere

This is an optimization guideline, not a hard requirement for the first V2 pass.

## Non-Goals

Version 2 still does not include:

- learned boundary scoring
- pose-sequence modeling
- serve classification models
- shuttle tracking
- global video-level boundary ranking

The purpose of Version 2 is to make the heuristic signal system real before escalating complexity.

## Validation Expectations

Version 2 should be judged by two things together:

### 1. Signal-decision metrics
The refinement pass should show a clear reduction in fallback dependence.

### 2. Human review
Old/new boundary pairs should still be reviewed by hand on a small sample set, with attention to:

- complete-rally feeling
- more natural starts
- more natural ends
- no visible bleed into the next rally

## Version 2 Completion Criteria

Version 2 is considered successful when:

- boundary reasons truthfully distinguish signal-matched vs fallback cases
- fallback rates are measurable and meaningfully reduced versus the current skeleton behavior
- end/start decisions use actual local signal transitions instead of fixed expansion alone
- next-serve protection remains safe and explicit
- human review shows better naturalness without obvious next-rally leakage

## Design Summary

- Keep the current architecture
- Upgrade `refine_boundaries()` from fallback-first to signal-first
- Use local state transitions, not global complexity
- Treat `valid_hits` as anchor constraints
- Treat motion as primary and audio as supporting signal
- Measure refinement quality by reduced fallback dependence, not just changed timestamps

## Implementation validation notes

- Track `start_fallback_rate`, `end_fallback_rate`, and `full_fallback_rate`
- Record `signal_matched_both`, `signal_matched_start_only`, and `signal_matched_end_only`
- Compare signal-first results with fallback-only behavior on a representative sample set

## Approval Gate

If this design is approved, the next step is to write an implementation plan that translates the start/end signal heuristics, reason codes, status fields, metrics, and next-serve contract into exact file changes and validation steps.
