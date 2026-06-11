# Best-possible 2D gaze from `calibration/test1.mp4` (raster SLO) — study verdict

**Question.** Recover the best-possible 2D gaze trajectory `(perp(t), along(t))`
from the test1 raster SLO video, at **>1 kHz** and high accuracy. If that is not
possible, instead reconstruct a scalar attention metric `a(t)` for appending to
robot-teleop imitation-learning data.

**Headline answer — the >1 kHz goal is achieved, with margin.** test1 is a raster
capture (808 slow-axis columns × 1000 fast-axis rows, 1025 frames @ 14.633 fps).
Each frame is painted column-by-column over ~68 ms, so a *strip* of `S` adjacent
columns is a genuine 2D retinal snapshot taken at a sub-frame instant.
2D-registering strips to a reference (the **TSLO / Sheehy–Roorda** method,
Biomed. Opt. Express 3(10):2611, 2012 — 960 Hz at 0.66′ in their AOSLO)
recovers gaze at **strip rate = (808 / S) × 14.633 Hz**, i.e. **366 Hz → 11.8 kHz**
as `S` goes 32 → 1, far above the 14.6 Hz frame rate. Crucially, a 2D patch match
is **unique** — there is none of the 1D line-scan perp aliasing that forced the
earlier differential particle filter — so the method tracks **through saccades**
at arcmin accuracy, which the 1D pipeline could not.

`a(t)` is also delivered (below) as a bonus auxiliary signal, not a fallback.

---

## 1. Method

- **Workhorse:** incremental strip registration (`raster_track.py`, ref_mode
  `incremental`): each frame's strips are 2D NCC-registered to the previous
  frame; per-frame motion = median strip shift, within-frame residual =
  strip − median → the sub-frame detail. Relative (validated by detrended
  correlation), but robust to the large pursuit excursion + right-eye temporal
  FOV loss that corrupt an absolute mosaic on this capture.
- **Axes:** `perp` = fast axis (vertical gaze, the precise axis); `along` = slow
  axis (horizontal gaze, the strong-signal but coarser axis).
- **Scale (px→arcmin), measured per-phase vs the independent 32.5 Hz machine
  tracker:** along **0.584′/px** (r=0.62), perp **0.403′/px** (r=0.65). (Tracker
  is a noisy reference, so absolute arcmin carries ≈±30%; relative/rate trends
  and the certified synthetic numbers are scale-robust.)

## 2. Real-data rate→accuracy frontier (`raster_rate_accuracy.{png,md}`)

Per-stimulus-phase tracking fidelity (Pearson r vs the pursuit dot) and a
**truth-free precision floor** (RMS of gaze high-passed >40 Hz, where the eye has
≈no power, so it is dominated by per-strip registration noise):

| S | rate (Hz) | in-FOV % | r Hx | r Vy | r Cx | r Cy | r Lx | r Ly | NF perp (′) | NF along (′) |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 366  | 71 | 0.96 | 0.92 | 0.96 | 0.87 | 0.93 | 0.68 | 0.84 | 3.18 |
| 16 | 732  | 72 | 0.95 | 0.91 | 0.95 | 0.85 | 0.93 | 0.66 | 1.12 | 4.46 |
|  8 | 1478 | 73 | 0.95 | 0.92 | 0.96 | 0.85 | 0.93 | 0.65 | 1.35 | 5.07 |
|  4 | 2956 | 73 | 0.95 | 0.92 | 0.95 | 0.85 | 0.92 | 0.64 | 1.53 | 5.41 |
|  2 | 5912 | 74 | 0.95 | 0.92 | 0.95 | 0.85 | 0.91 | 0.64 | 1.75 | 5.76 |

(H/V/C/L = H_sine/V_sine/circle/lissajous; x = horizontal, y = vertical.)

**Findings.** (a) Tracking fidelity is essentially **rate-independent** from 366
Hz to 5.9 kHz: horizontal r ≈ 0.95–0.96, vertical r ≈ 0.85–0.92 (V_sine/circle).
Crossing 1 kHz costs nothing. (b) The precision floor grows only gently with
rate: vertical (fast axis) **sub-arcminute to ~1.8′**; horizontal (slow axis)
3–5.8′ (the weak axis — slow-axis position is set by fewer columns per strip).
(c) in-FOV ~71–74%, limited by right-eye temporal FOV loss on rightward gaze (a
hardware centration limit, not algorithmic — see `khz2d_vertical_diagnosis.md`).

## 3. Certified accuracy on labeled synthetic (`raster_synth_certified.{png,md}`)

Real test1 has no high-rate ground truth, so absolute accuracy is certified on a
synthetic raster rendered from a clean test1-derived retina + a **known** 2D gaze
(pursuit + OU fixational drift + ballistic saccades) at the real per-column
timing, with the measured per-rate noise, recovered against a perfect reference:

| S | rate (Hz) | lock % | FIX perp (′) | FIX along (′) | SAC perp (′) | SAC along (′) |
|---|---|---|---|---|---|---|
| 32 | 366   | 100 | 1.39 | 3.10 | 1.52 | 2.88 |
| 16 | 732   | 100 | 1.69 | 3.28 | 1.28 | 2.04 |
|  8 | 1478  | 100 | 1.79 | 3.59 | 1.32 | 2.44 |
|  4 | 2956  | 100 | 1.90 | 3.73 | 1.39 | 2.51 |
|  2 | 5912  | 100 | 1.97 | 3.81 | 1.36 | 2.42 |
|  1 | 11823 | 100 | 2.06 | 3.92 | 1.38 | 2.47 |

**Findings.** (a) **100% lock at every rate including the 11.8 kHz line rate** —
the 2D match never aliases. (b) Fixation/pursuit accuracy ≈ **1.4–2.1′ vertical,
3.1–3.9′ horizontal**, rate-robust to the line rate. (c) **Through-saccade
accuracy is equally good (1.3–1.5′ vertical, 2.0–2.9′ horizontal) at ALL rates** —
NOT blur-limited. This is the decisive contrast with the 1D line-scan DPF, whose
through-saccade error was 36–53′ (`rate_sweep_verdict.md`, G13). The certified
fixation numbers bracket the real-data >40 Hz noise floor (e.g. 1.79′ vs 1.35′
perp @1478 Hz), cross-validating the two independent measurements.

## 4. Recommended operating point

**S = 8 → 1478 Hz** is the sweet spot above 1 kHz: full fidelity (r 0.95 horiz /
0.92 vert), certified ≈1.8′ vertical / 3.6′ fixation, ≈1.3′/2.4′ through-saccade,
73% in-FOV. For maximum temporal resolution, **S = 2 → 5.9 kHz** or **S = 1 →
11.8 kHz** are viable with only a mild precision cost (perp floor 1.75′→2.1′).
Lower rates (S=16/32) buy a slightly tighter floor if ≤732 Hz suffices.

## 5. Scalar attention metric `a(t)` (`raster_attention.py`, `raster_attention.png`, `raster_attention_track.csv`)

Delivered as a bonus (the >1 kHz gaze goal was met, so this is additive, not a
fallback). Built on the raster strip-tracking gaze. (A complementary a(t) built
on the M4 line-scan reconstruction already exists in `attention.py` — an
intake/saccadic-suppression × lock gate; both are valid, sensor-diverse views.)

`a(t) ∈ [0,1]` (+ confidence `c(t)`) fuses four
sensor-diverse, literature-grounded correlates of attention/engagement, each
z-scored over the session, computed from the kHz gaze binned to a 240 Hz
kinematics grid (eye motion is band-limited; binning removes the per-strip noise):

1. **fixation/tracking stability** — residual gaze jitter after removing the
   <3 Hz pursuit component (so following a moving target is *not* penalized; only
   tremor/jitter/lag remains). Low = steady, focused.
2. **(micro)saccade inhibition** — negative windowed saccade-event rate
   (Engbert–Kliegl detection); sustained concentration suppresses saccadic
   sampling.
3. **pursuit gain** — windowed eye-vs-target velocity regression (target locked).
4. **pupil-linked arousal** — tonic + phasic pupil area from the machine tracker
   (an INDEPENDENT sensor; LC-NE arousal / cognitive effort, Kahneman / Mathôt).

**Validation (self-consistency — necessary, not sufficient; no ground-truth
attention label exists):** inter-component agreement r(stability, inhibition) =
**+0.57**; `a(t)` is higher in engaged (low-saccade-rate) epochs than in lapsing
epochs (**0.59 vs 0.41**); `a(t)` tracks the independent pupil sensor at
r = **+0.27**; 93% of samples valid. Confidence correctly collapses during
blink / FOV-loss epochs.

**Teleop export:** `export_for_teleop()` resamples `a(t)` + `c(t)` to the control
rate (default 100 Hz) → `results/raster_attention_track.csv` with schema
`t_s, attention, confidence`. Append `attention` (gated by `confidence`) to each
teleop trajectory as an auxiliary imitation-learning channel.

## 6. Honest limitations

- Absolute arcmin scale rests on the noisy 32.5 Hz tracker (r≈0.6) → ≈±30%;
  the rate trends, lock behaviour, and certified-synthetic accuracy are scale-robust.
- ~27% of strips are out-of-FOV (right-eye temporal gaze) and are masked, not
  tracked — a hardware FOV/centration limit. Re-centering the raster or a wider
  FOV would recover them.
- Horizontal (slow axis) precision (3–6′) is intrinsically coarser than vertical
  (<2′); the slow axis is the weak axis of this scanner.
- The certified synthetic uses a perfect reference + full coverage (best case);
  real data adds reference noise + FOV loss. The two agree to within the noise
  floor, but synthetic is the optimistic bound.
- `a(t)` self-consistency is necessary, not sufficient; on a pure-pursuit task it
  is anchored by pursuit gain + the independent pupil channel. Re-validate the
  component weights on representative teleop data.

## 7. Reproduce

```
python raster_study.py     --S 32 16 8 4 2     # real rate/accuracy frontier + fig
python raster_synth.py     --S 32 16 8 4 2 1   # labeled certified accuracy + fig
python raster_attention.py                     # a(t) metric, validation, teleop CSV
python raster_track.py     --S 8 --report      # single operating-point tracker
```

Artifacts: `results/raster_rate_accuracy.{png,md}`, `raster_synth_certified.{png,md}`,
`raster_attention.png`, `raster_attention_track.csv`, and this verdict.
