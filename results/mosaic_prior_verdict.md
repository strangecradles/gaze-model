# Belief-level mosaic motion prior — implemented, but insufficient for flip rejection

In-PF answer to "can we filter out the non-physiological mosaic-alias jumps inside the particle
filter?" A flag-gated, default-OFF, byte-identical belief-level prior was implemented and
empirically evaluated. **It does NOT meaningfully reduce the mosaic-alias flip-rate**, for an
architectural reason that is itself the useful finding. Per subject, never pooled.

## What was built (and is correct)

`filter.py` `mosaic_prior` (flag-gated, default OFF): during armed locked pursuit, multiply a soft
weight factor `w_mosaic = exp(-0.5*(excess/sigma)^2)` into the particle weights, where
`excess = max(0, |pos_perp - anchor| - pursuit_budget)` at the fine `MOSAIC_SPACING_ROWS=6` scale.
The anchor is a **slow EMA** (`tau≈3 ms`) of the point estimate, gated to relax in saccade mode
(`p_pursuit≥0.9`), at low NCC (dedicated `mosaic_prior_ncc_thr`), and for `reseed_hold` steps after
a reseed. **OFF is bit-identical** to the committed baseline (verified in-process and on the people
pipeline, `max|Δx_px|=0`); 6 unit tests + existing fixation/saccade/reacquire suites pass.

Two design fixes were forced by evidence during implementation:
1. the per-step budget must be the **pursuit** velocity, not `v_peak(MOSAIC)` (the prior only fires
   in pursuit; a saccade-speed budget made it inert);
2. the anchor must be a **slow EMA**, not the per-step estimate — a per-step anchor *follows* the
   gradual roughening migration and is blind to it.

## Power-check (Igor, 15 s segment; the metric of record)

| variant | flip-rate | core-RMS | microsacc-pres |
|---|---|---|---|
| OFF (baseline) | 0.413 | 1.060′ | 0.80 |
| per-step anchor, arm NCC 0.35 | 0.412 (Δ−0.001) | 0.997′ | 0.80 |
| per-step anchor, arm NCC 0.20 | 0.413 (Δ+0.001) | 0.973′ | 0.80 |
| **EMA anchor (τ3ms), arm NCC 0.20** | **0.408 (Δ−0.004)** | **0.948′ (−11%)** | 0.78 |

Flip-rate barely moves (≤1% relative). The consistent, real effect is a **~6–11% core-RMS
improvement** on clean/locked lines (the prior suppresses sub-mosaic jitter). *(Ashton3 power-check
pending; the mechanism below is subject-general.)*

## Why it can't reject the flips (the architectural finding)

Arming is **not** the bottleneck: at arm-NCC 0.20 the prior is armed at **92% of flip-prone lines**
(median flip-prone NCC 0.32). Yet the flips persist. The reason:

> A soft prior can only **bias existing** multimodality; it cannot **resurrect** a mode the cloud
> has already abandoned. By the time the flip line arrives, the single roughened particle cloud has
> **already collapsed onto the wrong mosaic peak** — through the gradual sub-per-step roughening
> migration and the wide reseed (`N(anchor, 125 rows)` sprays ~20 mosaic peaks) in the preceding
> sustained-low-NCC region. With **no true-peak particles left**, the prior down-weights the whole
> (wrong-peak) cloud uniformly and the EMA anchor eventually follows it.

This is consistent with every prior negative result (the flip lives in the cloud *position*; it is
a multi-line correlated excursion, not a single jump). The minimal zero-output-lag belief prior is
the wrong tool because the multimodality has already collapsed before it can act.

## Verdict + recommendation

- **Primary goal (flip rejection): not achieved** by the belief prior (Δ flip-rate ≤ −0.004). No
  full 2×16-min A/B was run — the power gate on the primary metric is not met.
- **Secondary benefit (real):** ~6–11% lower core-RMS on locked lines (needs full-build r-vs-dot
  confirmation that it is precision, not pursuit distortion).
- The code is **flag-gated default-OFF, byte-identical, tested** — safe to keep as the substrate for
  the escalation, or to revert.
- **What would actually work** (the bounded-lag escalation the user deferred): keep the competing
  mosaic peaks alive as **explicit hypotheses** and defer the commit by a small lag, resolving by
  persistence + main-sequence — i.e. preserve the true-peak mode through the ambiguous region
  instead of trying to recover it after collapse. This attacks the collapse at the source.

Artifacts: `mosaic_prior_ab.py`, caches `m4_mosaic_prior*_d15*.npz`, this verdict. Implementation:
`filter.py` (`mosaic_prior*`), `calib.py` (`MOSAIC_SPACING_ROWS`), `people_fov_pf.run_m4`,
`tests/test_mosaic_prior.py`.
