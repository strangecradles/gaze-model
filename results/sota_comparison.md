# SOTA head-to-head — Stevenson & Roorda composite-reference strip registration vs our particle filter (test1 raster)

A faithful implementation of the current state-of-the-art retinal image-based
eye tracker — **strip-based registration to a composite/synthetic reference
frame** (the TSLO/AOSLO method of Stevenson & Roorda; the refined 2020 Roorda-lab
"robust strip-based digital image registration"; substrip variant Liu et al.
2024) — run on OUR real `test1` pursuit raster and scored with the EXACT same
`khz2d.evaluate` protocol as every other method (2 ms smoothing, 0.05 Hz drift
removal, per-axis affine calibration to the 0.2 Hz pursuit dot, r vs the ~32.5 Hz
machine tracker, precision = RMS of >25 ms detail).

## The SOTA algorithm (as implemented, `sota_strip.py`)

1. **Pre-processing.** Per-frame CLAHE contrast enhancement + de-band; blink /
   low-intensity frame rejection; distortion-frame rejection via the
   consecutive-frame full-frame match quality (`khz2d.chain` `q`) below 0.45.
2. **Composite (synthetic) reference.** Accepted low-distortion frames are
   averaged at their globally-registered positions into an oversized composite
   (a retinal mosaic) — registration is to this FIXED composite, NOT to the
   previous frame. This is the defining difference from incremental tracking.
   (632/1025 frames entered the composite; coverage 91%.)
3. **Strip registration.** Each frame is split into strips of S adjacent columns
   (a strip is parallel to the fast scanner = a column on this raster). Each
   strip is NCC-matched (`TM_CCOEFF_NORMED`) to the composite within a ±60 px
   local window; sub-pixel peak by 2D parabolic interpolation; accepted iff the
   NCC peak exceeds a threshold.
4. **High-rate trace.** Per-strip (x, y) offsets in temporal order give the eye
   trace at (808/S)·14.633 Hz: S=8 → 1478 Hz, S=1 → 11823 Hz (per column).

**Citations.** Stevenson & Roorda, Proc. SPIE 5688 (2005); Sheehy et al.,
Biomed. Opt. Express 3(10):2611 (2012); Bowers/Boehm/Roorda robust strip
registration (BOE, 2019–2020); Liu et al. substrip variant (2024).

## Implementation notes & honest simplifications

- **Global coordinate / reference selection.** The composite is built in the
  coordinate of `khz2d.chain()` — the robust incremental full-frame (strip-median)
  registration that EVERY method already uses as its 20 Hz absolute anchor. This
  replaces manual reference-frame selection. It is **not a handicap**: it gives
  the SOTA the *same* coarse anchor as M1/M4, so the comparison isolates the only
  thing under test — the high-rate per-strip residual estimator
  (composite-reference NCC vs previous-frame NCC vs particle filter).
- **Honest modality caveat.** The published sub-arcminute numbers are on
  cone-resolved AOSLO. On our video-rate NON-AO raster the composite is an
  averaged mosaic whose NCC peaks are intrinsically *lower* than a single sharp
  neighbour frame (median strip-NCC ≈ 0.31 to the composite). That lower SNR is
  exactly why "same data, same modality" is the only fair test — the SOTA is
  limited here by the same speckle/SNR as our methods.
- **Threshold / validity matching.** We report SOTA twice: at the *same numeric*
  strip-NCC acceptance as our incremental M1 (0.35), and at a *validity-matched*
  threshold (S=8 thr=0.20; S=1 thr=0.36) chosen so SOTA's valid% ≈
  the comparator's, so the precision number is computed on a comparable fraction
  of samples (a flat/over-rejected trace games precision — see guardrail below).
- **Compute.** S=8 over all 1025 frames (full 70 s). S=1 (per-column, 11823 Hz)
  over a 20 s window — the IDENTICAL window used for our published line-rate M4
  validation — because per-column NCC over the whole recording is heavy. Our M4
  numbers are reused from cache (the filter was not re-run).

## Guardrail (anti-gaming)

Precision is only meaningful if r_dot_x, r_trk_x and valid% are held. The
independent ~32.5 Hz tracker correlation `r_trk_x` is the honesty anchor;
`r_dot_x ≈ 0.9` is a pursuit-lag ceiling, not a target. We report all columns
together and do not crown a winner on precision alone.


## Head-to-head — ~1.5 kHz tier (full 70 s)

| method | rate (Hz) | r_dot_x | r_trk_x | RMS_x (′) | prec_x (′) | prec_y (′) | valid | note |
|---|---|---|---|---|---|---|---|---|
| SOTA composite S=8 (thr0.35) | 1478 | 0.820 | 0.537 | 18.7 | 3.60 | 3.02 | 38% | faithful thr |
| SOTA composite S=8 (valid-matched) | 1478 | 0.776 | 0.562 | 22.4 | 3.00 | 2.34 | 71% | thr=0.20 |
| Incremental strips M1 S=8 | 1478 | 0.878 | 0.642 | 19.8 | 4.39 | 1.25 | 72% | prev-frame ref |
| OUR best PF (learned) @1182 | 1182 | 0.905 | 0.547 | 16.2 | 1.90 | 0.98 | 71% | M4 best |
| OUR physics M4 @1182 | 1182 | 0.901 | 0.550 | 16.5 | 2.05 | 0.95 | 70% | M4 baseline |

## Head-to-head — 11.8 kHz line-rate tier (20 s window)

| method | rate (Hz) | r_dot_x | r_trk_x | RMS_x (′) | prec_x (′) | prec_y (′) | valid | note |
|---|---|---|---|---|---|---|---|---|
| SOTA composite S=1 (thr0.35) | 11823 | 0.838 | 0.501 | 19.3 | 4.37 | 0.00 | 61% | faithful thr |
| SOTA composite S=1 (valid-matched) | 11823 | 0.840 | 0.501 | 19.1 | 4.38 | 0.00 | 60% | thr=0.36 |
| Incremental strips M1 S=1 | 11823 | 0.850 | 0.651 | 25.3 | 5.94 | 0.00 | 66% | prev-frame ref |
| OUR best PF (learned) @11823 | 11823 | 0.796 | 0.559 | 23.2 | 1.57 | 0.00 | 60% | M4 best |
| OUR physics M4 @11823 | 11823 | 0.797 | 0.553 | 23.2 | 1.66 | 0.00 | 60% | M4 baseline |

## Verdict — what our method buys over the SOTA, on our data

- **~1.5 kHz / 1182 Hz:** OUR PF precision_x **1.90′** vs SOTA **3.00′** at matched validity (**Δ = +1.11′**, +37% — PF better). r_trk_x PF 0.547 vs SOTA 0.562 (Δ -0.014). r_dot_x PF 0.905 vs SOTA 0.776. valid PF 71% vs SOTA 71%.
  - Composite reference vs incremental strips: SOTA 3.00′ vs M1 4.39′ (**+1.39′** — composite better; validates the SOTA's key idea).
- **11.8 kHz line rate (20 s):** OUR PF precision_x **1.57′** vs SOTA **4.38′** at matched validity (**Δ = +2.81′**, +64% — PF better). r_trk_x PF 0.559 vs SOTA 0.501 (Δ +0.058). r_dot_x PF 0.796 vs SOTA 0.840. valid PF 60% vs SOTA 60%.
