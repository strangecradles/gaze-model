# Phase A (H4) — duration discriminator REFUTED, redirected to velocity/slew-rate

Per subject, never pooled. Metric on the PF output + harness rdx. Cross-reference labels are
OFFLINE validation only.

## H4 (duration separates flips from microsaccades) — REFUTED

Excursion reversion-duration of ≥3px departures, by cross-reference label:

| | flip median (lines) | "microsacc" median | flips persist >5.5ms | best L separation |
|---|---|---|---|---|
| Igor | 7–11 | 18–21 | 14–20% | +0.13 (weak) |
| Ashton3 | 8–12 | 15–28 | 21–25% | +0.26 (weak) |

The distributions **overlap** — flips are NOT 1–3 line transients (median 7–11 lines) and
14–25% persist beyond 5.5 ms like steps. At every candidate lag L, rejecting flips also
"clips" a comparable fraction (0.4–0.68) of the cross-reference-"consistent" set. By the
objective's own gate (overlap → refute), H4-as-duration is refuted.

## The reframing finding (load-bearing)

The cross-reference "≥3px-jump = microsaccade" label captures **ZERO real microsaccades**:

- **0%** of cross-reference-consistent ≥3px jumps persist ≥88 lines (6 ms, the real-microsaccade
  minimum) — on both the PF output and the harness rdx.
- Their peak per-line velocity is **119–419 °/s**, vs ~a few °/s physiological for a 0.04°
  (2.86′) amplitude. **Non-physiological.**

Real microsaccades are not ≥3px *jumps* at all — they are **smooth low-velocity ramps** over
~146 lines (10 ms). Therefore **every ≥3px per-line jump in this signal is a flip / noise
transient**, and the "duration overlap" is between two *noise* populations (the cross-ref label
never contained real microsaccades).

## Redirect: VELOCITY / slew-rate is the clean discriminator (near-zero lag)

Because real movements are smooth sub-threshold ramps and flips are non-physiological jumps,
the separating feature is **per-line velocity**, not duration:

- flip onset/revert: ~6px in 1 line ≈ **>100 °/s** (alias jump)
- real microsaccade / pursuit: smooth ramp **< ~50 °/s** per line

A per-line **slew-rate clamp / velocity gate** on the readout rejects non-physiological jumps at
**near-zero lag (1–2 lines, ~0.1 ms)** and **cannot clip real microsaccades** (they never exceed
the physiological per-line velocity). This is the discriminator the prior loop never tested — it
killed NCC / prediction-distance / mode (which overlap), but per-line velocity does not overlap.

It also explains why the lock-gated and ungated window mechanisms failed: they operated on the
posterior *position*, not on the *non-physiological velocity* of the jump that produced it.

## Verdict + next step (Phase B redirect)

H4 (duration, multi-line lag) is refuted, but its *spirit* (bounded-lag online rejection) is
correct with the right feature: **velocity/slew-rate (H5), at near-zero lag.** Phase B should:

1. Implement a flag-gated per-line slew-rate clamp on the PF output (default OFF, byte-identical):
   a position change exceeding the physiological max per-line velocity is held/limited to the
   prior estimate until a physiological ramp confirms a real move.
2. Power-gate it; sweep the velocity threshold; report flip-rate reduction and **latency (≈0)**.
3. Signal-safety: validated by **r-vs-dot (smooth-pursuit fidelity)** and ramp preservation —
   NOT by ≥3px-jump "microsaccade" counts (which contain no real microsaccades here).

Residual after a velocity gate: the **persistent flips** (14–25%, sustained wrong-peak with
physiological-looking plateaus) are not caught by velocity alone and remain the irreducible
online floor; the rest (~75–85% of flips, the jump-and-revert ones) are rejectable at ~0 lag.

Artifacts: `duration_split.json` (per-subject duration histograms, velocities, persistence).
