# Boundary Refinement V2.1 Truthfulness and Observability Design

**Date:** 2026-04-17
**Scope:** Follow-up design iteration for Boundary Refinement V2, focused on making signal-first behavior truthful, measurable, and aligned with the current implementation claims.

## Goal

Make the signal-first refinement system trustworthy before making it more aggressive. Version 2.1 should prioritize truthful decision reporting, meaningful fallback metrics, and real use of all declared decision inputs over more ambitious boundary movement.

## Primary Product Direction

Version 2.1 is not mainly about “stronger-looking refinement.” It is about making the system answer these questions honestly:

- Did audio actually contribute to this decision?
- Was this boundary truly signal-matched or just a fallback?
- Was the final result preserved or overridden by safety rules?
- Are fallback rates actually decreasing, or do we only think they are?

This version therefore prioritizes **trustworthy observability over aggressive behavior change**.

## Why This Version Exists

Version 2 moved the system much closer to real signal-first refinement:

- start and end boundaries now attempt local transition-based anchors
- explicit reason codes exist
- refinement status exists
- aggregate metrics exist

But the current implementation still has credibility gaps:

- `energy_points` are present but not used in decisions
- `signal_match_min_score` exists but does not gate anything
- `refinement_status` does not fully express next-serve protection overrides
- fallback reporting is not complete enough to measure true dependency

So Version 2.1 is the “make the claims true” release.

## High-Level Strategy

Keep the V2 architecture and rules mostly intact, but make four changes:

1. make audio participate in candidate validation
2. make `signal_match_min_score` actually decide whether a candidate is accepted
3. make final status reflect the real final boundary source
4. make fallback and unchanged behavior measurable in a way that matches the design intent

This version should not introduce a more complex scoring engine than necessary. It should minimally extend the current heuristic system so that current docs, metrics, and implementation are aligned.

## Decision Model Changes

### Current situation
The current V2 code does this:

- use motion to propose a candidate
- if a motion heuristic triggers, accept it immediately
- otherwise fallback

### Version 2.1 target
The system should instead do this:

1. use motion to propose a candidate anchor
2. evaluate whether local supporting evidence is strong enough
3. accept the candidate only if the evidence clears a confidence gate
4. otherwise fallback

This preserves the heuristic structure while making signal acceptance more honest.

## Audio Participation

### Role of audio in Version 2.1
Audio should remain a **supporting signal**, not a co-equal primary signal.

That means:

- motion remains the main source of candidate generation
- audio contributes to candidate validation
- a signal candidate is more credible when motion and audio agree
- a signal candidate is weaker when motion suggests an anchor but audio provides no local support

### Start boundary audio use
For start candidates, audio should help answer:

- does local energy also increase around the proposed start anchor?
- is the transition consistent with entering active rally behavior?

### End boundary audio use
For end candidates, audio should help answer:

- does local energy reduce or disappear around the proposed cooldown anchor?
- is there evidence that the rally stopped instead of continuing noisily?

### Why this is enough
This keeps audio useful without requiring the system to identify exact serve sounds or learned acoustic classes.

## Signal Confidence Gate

### Problem
The current system treats “heuristic found a candidate” as equivalent to “candidate is trustworthy.”

That is too permissive.

### Version 2.1 rule
Candidate acceptance must go through a confidence gate using:

- motion evidence
- audio support
- constraint consistency

### Minimal scoring structure
A lightweight additive score is enough for Version 2.1:

- `motion_score`
- `audio_score`
- `constraint_score`

Then:

- `total_score = motion_score + audio_score + constraint_score`
- accept the candidate only when `total_score >= signal_match_min_score`

This is not a general scoring engine redesign. It is a minimal honesty gate for the current heuristic system.

## Constraint Use of `valid_hits`

Version 2.1 should make `valid_hits` part of the real decision, not just nearby context.

### Role of `valid_hits`
- for start anchors: the candidate should plausibly lead into the first hit cluster
- for end anchors: the candidate should plausibly follow the last hit cluster

### Practical effect
A candidate that is too temporally disconnected from the valid hit cluster should receive lower confidence or be rejected.

This strengthens the “signal-first” claim without requiring a more complicated sequence model.

## Truthful Final Status

### Problem
A boundary may be signal-matched first and then overridden by next-serve protection. If the status does not reflect that, the system is reporting the decision history, not the decision outcome.

### Version 2.1 principle
`refinement_status` must describe the **final boundary outcome**, not a temporary intermediate state.

### Required status behavior
If the final exported end boundary is reduced by next-serve protection, the rally-level status should reflect that override explicitly.

Recommended values:

- `signal_matched_both`
- `signal_matched_start_only`
- `signal_matched_end_only`
- `fallback_both`
- `clamped_by_next_serve`

This is intentionally simple. The purpose is clarity, not exhaustiveness.

## Reason Field Semantics

Per-side reasons should continue to reflect local explanations, but they must stay truthful.

### Start reasons
- `signal_matched_motion_rise`
- `signal_matched_low_activity_exit`
- `fallback_front_pad`

### End reasons
- `signal_matched_motion_cooldown`
- `signal_matched_low_activity_entry`
- `next_serve_protection`
- `fallback_back_pad`

A signal reason should only appear if a candidate cleared the confidence gate.

## Metrics That Must Become Real

Version 2.1 should explicitly add or complete these metrics:

- `start_fallback_rate`
- `end_fallback_rate`
- `full_fallback_rate`
- `unchanged_count`
- `signal_refined_count`
- `full_signal_refined_count`
- `partial_signal_refined_count`
- `fallback_refined_count`
- `next_serve_clamped_count`

### Meaning
These are not vanity counters. They are how the system proves whether it is actually becoming less dependent on fallback logic.

## `unchanged_count` Semantics

A rally should count as unchanged when refinement leaves the final boundary effectively the same as the rough boundary.

This is useful because:

- it prevents overinterpreting any movement as improvement
- it shows when rough boundaries were already good enough
- it distinguishes “signal failed” from “signal decided no meaningful change was needed”

## Next-Serve Protection Contract

Version 2.1 keeps the V2 contract:

- next-serve protection is based on the next rally’s `raw_start_time`

That contract is still correct and should not be changed here.

What Version 2.1 changes is only the truthfulness of how the override is reported in status and metrics.

## Validation Priorities

Version 2.1 validation should focus on trustworthiness first.

### Required validation questions
- Did audio actually participate in candidate acceptance?
- Did the confidence gate reject weak candidates that V2 would have accepted?
- Do status and reason fields reflect final output truthfully?
- Do fallback metrics now directly answer how dependent refinement still is on defaults?

### Human review still matters
The old/new clip comparison still matters, but Version 2.1 does not need to chase dramatic visible gains first. It needs to make later gains measurable and believable.

## Non-Goals

Version 2.1 does **not** aim to:

- redesign the whole boundary heuristic system
- replace heuristics with a learned score model
- introduce richer event-classification features
- make boundaries substantially more aggressive by default

Those belong to later refinement iterations.

## Completion Criteria

Version 2.1 is successful when:

- `energy_points` are genuinely used in candidate validation
- `signal_match_min_score` genuinely gates candidate acceptance
- `refinement_status` truthfully reflects next-serve clamping outcomes
- fallback metrics and unchanged counts are complete and trustworthy
- docs no longer claim behavior the implementation does not actually perform

## Design Summary

- Keep V2 architecture unchanged
- Add audio as a supporting decision signal
- Add a real confidence gate for signal acceptance
- Make final status and reasons truthful
- Make fallback dependency measurable in a way that matches actual output behavior
- Optimize for trustworthiness and observability before more aggressive rule tuning

## Implementation validation notes

- Verify that audio affects candidate acceptance, not just metadata flow
- Verify that `signal_match_min_score` changes acceptance behavior
- Verify that next-serve clamping changes `refinement_status` to `clamped_by_next_serve`
- Verify that fallback and unchanged metrics are populated per video

## Approval Gate

If this design is approved, the next step is to write an implementation plan translating these trustworthiness changes into exact code updates, metrics definitions, and validation steps.
