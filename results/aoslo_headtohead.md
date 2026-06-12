# AO-SLO head-to-head: our particle filter vs. the SOTA strip-registration method

**Question.** Does our multimodal differentiable particle filter (DPF) beat the
state-of-the-art strip-registration method in **absolute** eye-position accuracy
at the AO-SLO (cone-resolved) scale — the regime where aliasing is sharpest and
our roadmap predicts the biggest multimodal advantage?

**Answer (headline).** On the public UC-Davis simulated AO-SLO set *with ground
truth*, **our DPF more than halves the absolute RMS error of the SOTA**:
0.055 arcmin (DPF) vs. 0.123 arcmin (SOTA, sub-pixel) / 0.155 arcmin (SOTA,
integer-lag, as the original script reports). This is measured on the SOTA
authors' own data, on their own hardware's image format, against their own
ground-truth eye traces. On real frames (no GT) the two methods agree well
(r = 0.94 horizontal, 0.88 vertical); the honest trade-off there is that our
single-row DPF is noisier than the SOTA's 13-row strip average (see below).

---

## Dataset and citation

- **Code/data:** R. Jonnal, *intraframe_motion_correction*
  (https://github.com/rjonnal/intraframe_motion_correction), UC Davis Vision
  Science & Advanced Retinal Imaging (VSRI).
- **Method/paper (SOTA baseline):** Azimipour et al., *"Intraframe motion
  correction for raster-scanned adaptive optics images using strip-based
  cross-correlation lag biases,"* PLoS One, 2018.
- **Simulated motion model:** Engbert et al., PNAS 2011 (self-avoiding-walk
  fixational drift). Frames built by the repo's `create_simulated_images.py`.

The repo was cloned into `external/intraframe_motion_correction/` (added to
`.gitignore`; **not** committed into our tree).

### What's in the data (verified, not assumed)

| Item | Verified contents |
|---|---|
| `object/full_mosaic.npy` | 512×512 simulated retinal cone mosaic (the clean "atlas") |
| `slo_frames_simulated/NNN.npy` | **200** frames, **128×128**, float64, 30 Hz, motion-affected |
| `slo_frames_simulated/resources/eye_trace_{x,y}.npy` | **(200,128)** per-line ground-truth eye trace — this is the **actual** GT used to render the frames (= 1.5× the raw `simulated_eye_traces/`, confirmed by reconstructing frames bit-exactly) |
| `slo_frames_simulated/resources/motion_free.npy` | 128×128 = `mosaic[100:228,100:228]` (confirmed `allclose`) |
| `slo_frames_real_large/NNN.npy` | **100** real UC-Davis AO-SLO frames, **512×512** |
| `demonstrate_registration.py` | the SOTA strip registration (Python **2**) |

### Scale and timing (stated explicitly)

- **Pixel scale:** each image subtends 2° over the 512-px mosaic sampling, so
  **deg/px = 2/512 = 0.00390625**, i.e. **0.234375 arcmin/px**. The 128-px
  simulated frame is a 0.5° window of that same mosaic, so the per-pixel angular
  scale is identical for simulated and real. All errors below are reported in
  **arcmin** using this scale.
- **Timing:** 30 Hz × 128 lines = **3840 lines/s**, dt = 0.26 ms/line.
- **GT motion:** fixational drift+tremor only (self-avoiding walk), ≈ ±4.6 px
  (±1.1 arcmin) peak-to-peak per frame, per-line jitter ≈ 0.33 px. **No real
  saccades** in this dataset, so the fixation/saccade split degenerates to
  all-fixation; the IMM's OU drift mode dominates.

### Geometry adaptation (the key engineering)

AO-SLO fast scan is **horizontal** (each acquired line is one full row, sampled
densely across 128 px); the raster advances **down** one row per line. This is
the *opposite* mapping from our column-raster harness, so we wired the filter as:

- **ALONG = horizontal (x, mosaic columns)** — trusted/dense (a full line gives a
  well-posed 1-D registration), fed as the tight `along_meas` (1-D normalized
  cross-correlation of the observed line vs. the atlas row).
- **PERP = vertical (y, mosaic rows)** — aliased/slow (only **one** row per time
  step → a 1-px-tall strip must be localized against the 2-D cone lattice, the
  multimodal regime). This is the axis carried by the particle cloud.

We verified the generative model exactly: `decoder.render(perp=100+idx+gy,
along=100+gx, full_mosaic, 128)` reproduces each stored simulated line to
**1.8e-15**. We track the eye **residual** (raster ramp removed) by feeding the
filter a per-line atlas window centred on the nominal raster row, so the perp
state is the eye offset (mirrors `khz2d_methods.m4_dpf`). Dynamics units were
recalibrated from our instrument's 124.6 rows/deg to this dataset's 256 px/deg,
and the OU velocity to the measured drift; **physics likelihood only** (the
learned head is trained on our synthetic data and does not transfer — noted, not
used).

---

## Step 1 — SOTA on its home turf (reproduced)

We ran the Azimipour strip registration (a faithful **Python-3 port** of
`demonstrate_registration.py`, since the original is Python 2 — equations 1–2:
per-strip FFT2 cross-correlation against the reference with the script's
horizontal-strip mean-bias removal, `strip_width = 13`). For the simulated set we
register each frame's strips against the **motion-free object** so each per-strip
lag is an *absolute* eye-position estimate, directly comparable to the GT trace.

**SOTA absolute accuracy (RMS vs. GT, all 200 frames, arcmin):**

| variant | x (along) | y (perp) | 2-D |
|---|---|---|---|
| integer lag (as the script reports) | 0.110 | 0.109 | 0.155 |
| + sub-pixel parabolic peak (fair best-case) | 0.087 | 0.087 | 0.123 |

The integer-lag floor (~0.45 px) is the cross-correlation quantization; adding
the standard parabolic sub-pixel peak fit (which strip-registration pipelines
normally use) gives the fairer 0.123 arcmin. Correlation with GT ≈ 0.93/0.93.

## Step 2 — our DPF, adapted to this data

One strip (one row) per filter step, physics likelihood, 250 particles,
per-frame run over the 128 lines (atlas = the clean `object/` mosaic, as allowed;
the SOTA uses the same object as its motion-free reference).

## Step 3 — the head-to-head (decisive, on the GT set)

**Absolute accuracy — RMS vs. ground truth, all 200 simulated frames, matched
strip rate (one estimate per row):**

| method | x (along) arcmin | y (perp, **aliased**) arcmin | 2-D arcmin |
|---|---|---|---|
| SOTA strip-reg (integer lag) | 0.110 | 0.109 | 0.155 |
| SOTA strip-reg (sub-pixel) | 0.087 | 0.087 | 0.123 |
| **our DPF (physics)** | **0.039** | **0.040** | **0.055** |

- **DPF beats the fair sub-pixel SOTA by 2.2× (0.055 vs 0.123 arcmin 2-D)**, and
  the as-published integer-lag SOTA by 2.8×.
- The win holds on **both** axes, and notably on the **vertical (perp) aliased
  axis** — the cone-resolved regime our roadmap predicted as our biggest edge.
- *Why we win:* analysis-by-synthesis renders a sub-pixel line directly from the
  atlas and the multimodal cloud resolves the cone-lattice ambiguity, whereas the
  strip method (i) is quantized to integer lags and (ii) must average ~13 rows for
  cross-correlation SNR, which **smooths out the fast tremor** the DPF tracks
  per-row (clearly visible in the figure's trace panels).

(See `results/aoslo_headtohead.png`.)

### Real frames (no GT): precision + qualitative agreement

12 real 512×512 frames, registered to frame 0 as reference/atlas:

| precision/agreement metric (arcmin) | SOTA strip-reg | our DPF |
|---|---|---|
| jitter — median abs 2nd-difference, x | **0.014** | 0.088 |
| jitter — median abs 2nd-difference, y | **0.008** | 0.035 |

Inter-method agreement (the two methods recover the *same* real trajectory):
horizontal RMS difference 0.169 arcmin (r = 0.943), vertical RMS difference
0.467 arcmin (r = 0.877). The vertical (aliased perp) axis is where they diverge
most — expected, since that is the hard axis.

The two methods **recover the same trajectories** on real data (r = 0.94 / 0.88;
see `results/aoslo_headtohead_real.png`). The **honest** trade-off: on noisy real
single rows, the SOTA's 13-row strip averaging yields a *lower-jitter* (smoother)
trace, while our per-row DPF is noisier — part of that "jitter" is genuine fast
motion the DPF resolves and the strip average suppresses, but part is real
estimation noise from working off a single, noisy, motion-distorted reference
row. We do **not** claim a precision win on real frames.

---

## Verdict (honest)

- **Absolute accuracy at the cone-resolved AO-SLO scale: WE WIN, decisively** —
  0.055 vs 0.123 arcmin 2-D RMS against ground truth (2.2× lower error), on the
  SOTA authors' own simulated data, including on the aliased vertical axis. This
  is a clean, publishable result *in our favour* on someone else's hardware/data.
- **Where the win is real vs. caveated:** the simulated frames are **noiseless**
  pure bilinear samples of a **clean** mosaic, and both methods use that clean
  object — so this measures the *algorithmic* ceiling (sub-pixel multimodal
  synthesis vs. integer-lag strip xcorr), where our method is genuinely better.
  Real-world photon/speckle noise will narrow the gap; our real-frame jitter being
  higher than the strip average is the concrete sign of that. The fair next step
  (noted, not done here) is to add calibrated noise to the simulated frames and/or
  retrain a likelihood head on their simulated set, clearly labelled.

## Reproduce

```bash
# clone the dataset (already in external/, gitignored)
git clone https://github.com/rjonnal/intraframe_motion_correction \
    external/intraframe_motion_correction
python aoslo_headtohead.py            # uses caches in results/aoslo_cache/
python aoslo_headtohead.py --rebuild  # recompute everything
```

**Files:** `aoslo_headtohead.py` (adapter + driver),
`results/aoslo_headtohead.png` (simulated GT head-to-head),
`results/aoslo_headtohead_real.png` (real-frame trajectories),
`results/aoslo_cache/` (cached lags/traces).
