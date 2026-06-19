# Phase B/C verdict — the mosaic-alias floor is ONLINE-irreducible

Per subject, never pooled. Metric of record = flip-rate + core-RMS on the PF OUTPUT.

## Phase A recap (established)

The alias flips SURVIVE into the PF output: flip-rate harness rdx 0.52 → PF output 0.39/0.40
(only ~25% reduction). The PF's alias-robust weighted mean uses a 30′ window, far too coarse
for the 2.86′ mosaic flip, so it averages the flipped mode in (autopsy: 40% reject / 32%
averaged-in / 28% followed). Reducible = cross-reference-inconsistent flips; irreducible =
real microsaccades (~3% of FOV lines) + core-RMS (~0.5′).

## Phase B — mode-gated `--mosaic-reject` (implemented, power-gated, FAILED)

Implemented in `filter.py` (flag-gated, default OFF **byte-identical** — verified fresh OFF vs
committed max|Δx_px| = 0.00e+00; 10/10 filter tests + 4 new unit tests pass). Mechanism (a):
when steadily locked, shrink the alias-robust-mean window from 30′ toward the mosaic scale
(~3 rows), keeping the wide window in saccade/lock-loss so real microsaccades survive.

**Power gate FAILED — and the diagnosis is conclusive:**

| run | flip-rate | core-RMS | note |
|---|---|---|---|
| OFF (committed) | 0.413 | 1.06′ | baseline |
| ON lock-gated (w=3) | 0.415 | 1.06′ | Δflip +0.002 — near-identity |
| ON ungated, w=1 row | 0.434 | 1.25′ | Δflip **+0.022 (WORSE)**, core +0.19, 37% lines changed |

1. **The flip lives in the cloud POSITION, not the posterior averaging.** Even an ungated,
   maximally aggressive 1-row window makes flip-rate *worse* (+0.022) and inflates core-RMS:
   the likelihood has already pulled the resampled cloud onto the flipped mosaic peak, so the
   MAP sits on the flip; narrowing the window just locks harder onto it. No window setting can
   reject a flip the cloud has already adopted. Mechanism (a) is dead.

2. **Flips occur where the lock gate is OFF.** Flip-prone lines have low NCC (median 0.32 <
   LOCK_NCC_THR 0.35; 41% locked) vs clean lines (0.59, 89% locked). Mode-gated rejection
   structurally cannot reach the flips without ungating — and ungating fails by (1).

## The online-discrimination wall (kills mechanism (b) and H3)

Real microsaccades and alias flips are **both ~mosaic-spacing departures from the causal
temporal prediction**, with **heavily overlapping NCC**:

| label | NCC median | NCC IQR |
|---|---|---|
| alias-flip-prone | 0.323 | [0.26, 0.42] |
| real-microsaccade | 0.424 | [0.32, 0.56] |
| clean | 0.593 | [0.45, 0.75] |

No causal/online feature (NCC, prediction-distance, mode posterior) separates them at the
moment of the jump. The only clean discriminator is **cross-reference consistency** (does the
line localize to the same place against DIFFERENT reference frames — alias: no, microsaccade:
yes) plus the main sequence — both inherently **OFFLINE / multi-frame**.

- **Mechanism (b)** (mosaic-scale prior penalty on the likelihood — the correct place, since
  the flip is in cloud position) is blocked by this wall: penalizing particles away from the
  prediction clips real microsaccades, which also depart from the prediction. Predicted to
  fail the mandatory signal-safety gate; not implemented (outcome determined by the overlap).
- **H3** (online local-search localizer) hits the identical wall (`h3.json`): NOT BUILT.

## Verdict + residual floor

The mosaic-alias floor is **online-irreducible within signal-safety**: any single-pass in-PF
or localizer rejection that lowers the alias-flip-rate clips real microsaccades by ~the
NCC-overlap fraction, violating the primary signal gate. No underpowered/unsafe A/B was run.

The only **signal-safe** lever is **OFFLINE multi-frame cross-reference averaging** (refavg3,
~11% reduction, already validated signal-safe via the staggered cross-reference coherence
test) — it cleans the reference using multi-frame information the online PF does not have.

Residual irreducible floor per subject (after the best signal-safe intervention, offline
refavg3): real-microsaccade content (~3% of lines, preserved) + core-RMS (~0.5′) + the
online-irreducible alias flips that only multi-frame averaging can partially remove.

Artifacts: `mosaic_power.json`, `h3.json`, this verdict; `mosaic_ab.py` (A/B harness, not run
— power gate failed); `tests/test_mosaic_reject.py` (4 pass). Default PF byte-identical (OFF).
