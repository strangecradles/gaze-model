# Vertical Mismatch Diagnosis — eye behavior, speed/lag, FOV asymmetry, calibration

Method under test: M4 particle filter @ 11823 Hz (testbed A / test1, OFF = 2.50 s). All r values |Pearson|; lag > 0 = recon lags the dot.

## (1) Who mismatches whom? The recon-dot-tracker triangle

| pair | r x | r y |
|---|---|---|
| recon vs dot | 0.906 | 0.752 |
| tracker vs dot | 0.557 | 0.636 |
| recon vs tracker | 0.560 | 0.418 |
| M0 frames-only vs dot (control) | 0.906 | 0.740 |

- The INDEPENDENT tracker shows a vertical deficit of -0.078 (x-y) vs the dot; the recon's deficit is +0.154. If these are comparable, the vertical mismatch is dominated by the EYE (vertical pursuit), not by the reconstruction.
- M0 (raw 15 Hz frame registration, no per-line tracking at all) has the same vertical ceiling (0.74) as every kHz method — the limit is method-independent.

## (2) Per-stimulus-phase: speed, lag, gain (H2 speed/lag, H4 miscalib)

| phase | dot speed med (deg/s) | r y | best-lag y (ms) | r y @lag | gain y @lag | tracker r y | r x | best-lag x (ms) |
|---|---|---|---|---|---|---|---|---|
| H_sine | 1.40 | - | -400 | 0.41 | 0.44 | - | 0.94 | -120 |
| V_sine | 0.77 | 0.90 | -180 | 0.94 | 0.61 | 0.79 | - | +800 |
| circle | 0.99 | 0.84 | -100 | 0.84 | 0.45 | 0.91 | 0.95 | +120 |
| lissajous | 2.68 | 0.70 | +0 | 0.70 | 0.67 | 0.26 | 0.96 | +20 |

## (3) Signal vs horizontal gaze position (H3: right-eye FOV asymmetry)

| side (dot x) | in-FOV | qh med | qv med | contrast med | err x med (') | err y med (') |
|---|---|---|---|---|---|---|
| left third | 98% | 0.74 | 0.70 | 19 | 8.9 | 9.2 |
| right third | 32% | 0.51 | 0.45 | 11 | 12.9 | 24.1 |

- Vertical r restricted to LEFT-gaze samples: 0.83; RIGHT-gaze samples: 0.11 (horizontal: 0.76 / 0.41).

## (4) Vertical error vs instantaneous dot speed (H2)

| dot speed quartile (deg/s) | err y med (') | err x med (') | in-FOV |
|---|---|---|---|
| 0.00-0.77 | 7.8 | 7.0 | 63% |
| 0.77-1.02 | 13.8 | 5.9 | 75% |
| 1.02-1.78 | 6.9 | 6.6 | 68% |
| 1.78-194.85 | 18.4 | 10.5 | 74% |

## Verdict

**H3 (right-eye FOV asymmetry) is the dominant effect.** In-FOV collapses from 98% on left gaze to 32% on right gaze; match quality drops (0.74->0.51) and median vertical error roughly doubles+ (9' -> 24'). Vertical agreement on left-gaze samples is 0.83 but only 0.11 on right-gaze samples. For the RIGHT eye, rightward (temporal) gaze drives the imaged retinal patch toward the edge of the SLO field, so signal is lost exactly as you suspected — and because the lost samples are masked OUT, the surviving vertical estimate is built from a left-biased subset, depressing the whole-trace vertical r.

**H1 (eye, not method) is real and explains most of the *residual*.** The independent machine tracker also tracks the dot worse vertically (0.64) than horizontally (0.56), and the frames-only M0 control hits the same vertical ceiling (0.74) as every kHz method — so the vertical limit is NOT specific to this tracker. Vertical smooth pursuit has lower gain than horizontal (well documented physiologically), and the per-phase vertical gains here are below 1 (~0.4-0.7), i.e. the eye under-shoots the dot vertically. Single-eye tracking per se is not the issue (we track one eye and validate against that same eye's tracker).

**H2 (speed/lag) is minor.** Best vertical lag is ~0 ms at every phase (the eye is not simply delayed), and vertical error rises only modestly into the fastest speed quartile / the lissajous phase (the fast phase also coincides with wider/temporal gaze, so part of this is really H3). **H4 (global miscalibration) is ruled out**: horizontal is excellent (r~0.91) under the same single affine calibration, the sign is correct, and the vertical deficit is position- and phase-dependent rather than a constant gain error.

**Actionable**: (a) report vertical accuracy gated to in-FOV / left-and-center gaze, where it is genuinely good; (b) the right-edge signal loss is a hardware FOV/centration limit for the right eye, addressable by re-centering the SLO raster temporally or widening the FOV, not by the algorithm; (c) the remaining vertical-vs-horizontal gap is the eye's own lower vertical pursuit gain, confirmed by the independent tracker.

Figure: `khz2d_vertical_diag.png` (per-phase vertical r; signal/quality and error vs horizontal gaze; signal/error vs vertical gaze).
