# Boundary Refinement for Competitive Rally Rhythm Design

**Date:** 2026-04-17
**Scope:** Accuracy-improvement phase focused on highlight boundary quality, specifically making exported clips preserve the rhythm of a complete competitive rally.

## Goal

Improve highlight clip boundaries so exported segments feel like完整竞技回合 rather than rough intervals with fixed padding. The new behavior should favor rally rhythm completeness over aggressive trimming, while avoiding a noticeable increase in false positives.

## Problem Statement

The current pipeline identifies rallies using valid hit sequences and then exports clips with fixed front/back padding. This works as a rough baseline, but clip boundaries often feel unnatural:

- Starts can be too early, pulling in idle setup and empty waiting time.
- Starts can be too late, cutting off serve preparation or the first meaningful movement.
- Ends can be too early, cutting the result of the final exchange.
- Ends can be too late, dragging in post-rally walking or next-serve preparation.

The root issue is that current boundaries are based on `first valid hit / last valid hit + fixed padding`, not on actual boundary anchor events in the rally timeline.

## Product Direction

This phase optimizes for **competitive rhythm completeness**:

- Start boundaries should preserve meaningful pre-rally buildup when it contributes to the sense of a complete point.
- End boundaries should preserve the result and cooldown of the final exchange.
- When forced to choose, the system should slightly prefer including a little extra context over clipping away rally rhythm.

This is intentionally different from a “cleanest trim possible” strategy.

## High-Level Approach

Add a dedicated **boundary refinement layer** between rough rally segmentation and clip export.

Current flow:

1. `segment_video()` finds rough rally intervals
2. `export_highlights()` applies fixed padding and writes clips

Proposed flow:

1. `segment_video()` finds rough rally intervals and preserves the event context needed for refinement
2. `refine_boundaries()` adjusts each rally’s `start_time` and `end_time` using local event alignment
3. `export_highlights()` exports the refined boundaries

This keeps responsibilities clean:

- `segment_video()` decides **whether a rally exists**
- `refine_boundaries()` decides **where that rally should begin and end for viewing quality**
- `export_highlights()` decides **how to cut and write the clip**

## Why a Dedicated Refinement Layer

Boundary quality is a distinct problem from rally detection.

Keeping refinement separate avoids three problems:

1. `hybrid_segmentation.py` becoming a mixed detector + presentation-logic file
2. `export_highlights.py` becoming a hidden decision engine instead of an output layer
3. Boundary tuning becoming hard to debug because rally detection and clip timing are entangled

A dedicated refinement layer makes future iteration easier and keeps the system explainable.

## Proposed Module Layout

### Existing modules
- `hybrid_segmentation.py`
  - Keep rough rally detection
  - Continue calculating `valid_hits`
  - Add enough intermediate event context to support refinement
- `export_highlights.py`
  - Keep clip cutting responsibility only
  - Use refined start/end times when present

### New module
- `boundary_refinement.py`
  - Input: rough rallies + event context
  - Output: refined rallies with explanation fields

## Boundary Model

Treat each rough rally as a candidate interval with two coarse anchors:

- `raw_start_time` = first meaningful valid-hit anchor for the rally
- `raw_end_time` = last meaningful valid-hit anchor for the rally

Then refine each side independently but asymmetrically:

- **Start boundary** = search for a pre-rally buildup anchor before `raw_start_time`
- **End boundary** = search for a post-rally cooldown anchor after `raw_end_time`

The rules are intentionally not symmetric, because start and end have different viewing semantics.

## Start Boundary Design

### Objective
Find the latest sensible point before the rally where the buildup into the first exchange still feels complete.

### First-version rule set
Given a rough rally start:

1. Search in a local window before `raw_start_time` (recommended first-version search window: about 1.5s to 2.5s backward)
2. Prefer the first strong candidate among:
   - onset of sustained motion increase before the first valid hit
   - end of the last stable low-activity region before the first valid hit
   - fallback front padding if no reliable event anchor is found

### Practical intent
The start should preserve serve/receive rhythm when it matters, but should not drift so far backward that it starts pulling in dead time from the previous point.

### Design bias
Because this phase prioritizes competitive rhythm, the start boundary should be slightly more permissive than a “tight highlight” strategy.

## End Boundary Design

### Objective
Find the earliest sensible point after the rally where the point feels complete, including result and immediate cooldown.

### First-version rule set
Given a rough rally end:

1. Search in a local window after `raw_end_time` (recommended first-version search window: about 1.0s to 2.5s forward)
2. Prefer the first strong candidate among:
   - first stable point where audio and motion both drop out of rally intensity
   - first low-activity point after the last strong exchange when no clear cooldown event exists
   - fallback back padding if no reliable event anchor is found

### Practical intent
The end should preserve score/result feeling and immediate rally completion, while preventing long drift into walking, pickup, or next-serve setup.

### Design bias
The end boundary should be slightly more conservative than the start boundary, because clipping the last exchange outcome hurts viewing quality more than keeping a small amount of extra tail.

## Signals to Use in Version 1

Version 1 should stay lightweight and use signals already available or cheap to expose from the current pipeline.

### Required signals
- `valid_hits`
- local audio energy changes near rally boundaries
- local vision velocity changes near rally boundaries
- transition between low-activity and high-activity periods

### Explicitly out of scope for Version 1
- pose-sequence classification
- serve action recognition models
- shuttle trajectory detection
- learned boundary scoring models
- point-outcome recognition

This keeps scope aligned with the current project maturity.

## Data Contracts

### Rough rally structure
The current rally data should be extended rather than replaced.

Recommended refined rally fields:

- `raw_start_time`
- `raw_end_time`
- `start_time`
- `end_time`
- `duration`
- `highlight_score`
- `boundary_adjustment_front`
- `boundary_adjustment_back`
- `boundary_reason_start`
- `boundary_reason_end`

### Intermediate event context
`segment_video()` should preserve or expose enough event context for refinement to avoid recomputing everything from scratch.

Recommended context for each video:

- `valid_hits`
- local or summarized audio-energy timeline usable near boundaries
- local or summarized vision-velocity timeline usable near boundaries

This can be emitted as an additional intermediate structure or passed directly in memory if the orchestration layer supports it.

## Safety Rules and Fallbacks

Boundary refinement must be an enhancer, not a destabilizer.

Mandatory fallback behavior:

- If no reliable start anchor is found, fall back to conservative front padding around `raw_start_time`
- If no reliable end anchor is found, fall back to conservative back padding around `raw_end_time`
- If a refined boundary would overlap the next rally’s protected region, clamp it to the safety limit
- If a refined boundary produces an obviously invalid duration, discard the refinement and keep the rough boundary behavior

This is a hard requirement for Version 1.

## Validation Plan

Boundary refinement cannot be judged only by “did it export a video.” It needs a small, focused quality evaluation.

### Evaluation style
Use **lightweight human evaluation plus supporting stats**.

### Sample set
Create a small set of representative rally clips that include:

- normal serve-start rallies
- fast exchanges with dense hits
- cases where current endings feel cut off
- cases where current endings drag too long
- known bad boundary examples from existing output

### Human review questions
For each old/new boundary pair:

1. Does the new start preserve the beginning of the point better?
2. Does the new end preserve the completion of the point better?
3. Does the clip feel more like a complete competitive rally?

### Supporting metrics
Track:

- average front adjustment size
- average back adjustment size
- fallback rate
- number of clips whose boundaries changed materially
- subjective rating for start quality, end quality, and overall naturalness

### Failure labeling
When the new boundary still feels wrong, classify the failure:

- missed buildup anchor
- missed cooldown anchor
- search window too wide
- search window too narrow
- false refinement due to non-rally motion

These labels should drive later iterations.

## Recommended First Implementation Scope

Version 1 should be intentionally modest:

1. Preserve existing rough rally detection
2. Add one new refinement pass that only searches in local windows around each rough boundary
3. Record raw vs refined times and reasons
4. Keep strong fallback behavior
5. Validate against a hand-picked sample set before broader rollout

This is enough to improve perceived clip quality without taking on a larger model or a major architecture rewrite.

## Non-Goals

This design does **not** try to solve:

- general rally classification from scratch
- perfect serve detection
- learned highlight ranking
- broadcast-quality editing semantics
- universal robustness across every camera angle and recording style in Version 1

Those may become later phases, but they are outside this design.

## Design Decision Summary

- Add `boundary_refinement.py` as a dedicated layer
- Optimize for complete competitive-rally rhythm, not shortest trims
- Use event alignment, not just fixed padding
- Keep rules local, interpretable, and fallback-safe
- Validate using lightweight human comparison rather than only pipeline success

## Implementation validation notes

- Compare old/new clip starts on a hand-picked rally sample
- Compare old/new clip ends on the same sample
- Track how many rallies changed boundaries materially
- Track fallback rate to rough-boundary behavior

## Approval Gate

If this design is approved, the next step is to turn it into an implementation plan covering:

- exact file changes
- data shape updates
- orchestration changes
- validation workflow
- staged rollout of boundary refinement version 1
