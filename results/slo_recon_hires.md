# High-resolution motion-corrected SLO reconstruction (test1) — troubleshooting + SOTA-style pipeline

**Question.** Why does the AO-SLO reconstruction (`results/aoslo_image_quality_real.png`)
look crisp while the first SLO reconstruction (`results/slo_recon_dpf_compare.png`)
ghosts and blurs — and what preprocessing/registration gets the SLO image as
close to SOTA (highest resolution) as the data allows?

## Diagnosis — why AO-SLO looked good and our first SLO didn't

**Intrinsic (unfixable without hardware).** The AO-SLO frames are *adaptive-optics
corrected*: a ~1° field, cone-level resolution, high SNR, and all 12 frames
fully overlap one tiny patch. Our `test1` capture is *conventional SLO* — no
adaptive optics, so a coarser eye-limited PSF, a wide (~30°) field, lower SNR,
and **no cones to resolve**. We can never reproduce the AO cone mosaic; "SOTA"
here means a clean, ghost-free, deconvolved register-and-average of conventional
SLO.

**Pipeline (fixable, and fixed below).** The first pass (`slo_recon_dpf.py`)
ghosted/blurred for four concrete reasons:

1. **Placement came from the incremental chain** (cumulative frame-to-frame
   registration). Cumulative registration drifts; a ~1 px systematic error over
   100 frames places vessels slightly differently each frame → **doubled/ghosted
   vessels**.
2. **Frames were averaged raw** — per-frame SLO scan banding + illumination
   falloff accumulate as **haze and stripes**.
3. **No frame/region quality gating** — blinks, low-contrast and FOV-dropout
   regions were averaged in → **blur**.
4. **The DPF gaze trace is tuned for gaze accuracy, not pixel-exact image
   registration.** It is the right tool for *tracking*; it is not a substitute
   for image-based registration when the goal is the sharpest *image*.

A key empirical find while fixing this: **register on raw frames, not
preprocessed ones.** Frame-to-frame NCC is ~0.50–0.56 on raw (z-scored) frames,
but de-banding drops it to ~0.3 and **CLAHE collapses it to ~0.1** (CLAHE
amplifies speckle; de-band removes the large-scale structure that drives
correlation). So flat-fielding/CLAHE must be deferred to the *final cosmetic*
step, never applied before registration.

## The SOTA-style pipeline (`slo_recon_hires.py`)

All on the same 0–20 s `test1` window (182 in-FOV frames), the way
Roorda/Azimipour register-and-average works:

1. **Register on raw frames** to an **iteratively-refined composite reference**
   (sub-pixel phase correlation; sharpest dwell frame → average → re-register,
   2 passes). Drift-free → **ghosting removed**. Median NCC 0.76 → **0.88**.
2. **Quality gating** — reject blinks (low de-banded energy) and frames whose
   registration NCC < 0.45; weight survivors by NCC. (All 182 frames survive in
   this window; **median averaging depth 182 frames/pixel** at the centre.)
3. **Dense 2D non-rigid refinement** — per-frame block-matching displacement
   field (tile 96, cubic-resized, smoothed). This removes **torsion + field
   warp** (median ~4 px) that 1-D strip translation cannot — the cause of the
   **wavy/doubled fine macular vessels**.
4. **Super-resolution** — shift-and-add onto a **2× finer grid** with cubic
   resampling; the eye's sub-pixel sample diversity resolves finer than one
   native pixel.
5. **Per-pixel quality weighting** — weight each frame's pixels by local
   contrast, so FOV-dropout / flat regions don't pollute the average.
6. **Cosmetic finish (after registration only)** — coverage-normalized
   flat-field, two-axis destripe (removes the residual vertical banding **and**
   the horizontal fast-axis scan line that survives because vertical gaze is
   stable), gentle Richardson-Lucy deconvolution, CLAHE display stretch.

## Result

See:
- `results/slo_recon_hires_before_after.png` — previous DPF recon vs the new
  pipeline (dramatically sharper, higher-contrast vessels; flat field; ~2× the
  pixels).
- `results/slo_recon_hires.png` — the 4-stage story: 1 raw frame → **unregistered
  mean (ghosted/blurred — the failure mode)** → image-registered + 2× SR
  (ghosting gone, speckle gone) → + deconvolution.
- `results/slo_recon_hires_hero.png` (+ `_native.png`) — the final image
  (1938 × 1615), sharp optic disc + full vascular arcades + fine branches.

The optic disc, arcades and most of the posterior pole are now sharp,
ghost-free, flat and continuous, at ~2× the native sampling. The biggest
remaining blemish is the **temporal macula**: low intrinsic contrast there, the
known right-eye **temporal FOV dropout**, and residual torsional warp leave some
mottling/waviness even after the dense refinement — this is a genuine
data-quality limit of conventional SLO in that region, not a pipeline bug.

## Honest caveats

- **SLO, not AO-SLO.** No adaptive optics → no cones, eye-limited resolution.
  The deconvolution recovers PSF-blurred detail, not diffraction-limited cells.
- **Reference-free.** No ground-truth retinal image; "sharpness" gradient
  numbers *fall* as N grows (speckle, which is high-gradient, averages out), so
  they are reported but not optimized — the deep-stack + dense-registration
  quality is judged visually + by ghost removal.
- **Registration, not the gaze trace, drives the image.** Per the goal of the
  *best image*, frame placement here is image-based (drift-free), which is what
  removes ghosting; the DPF gaze trace remains the right tool for the gaze
  product, and the two are consistent (both track the same motion).

## Reproduce

```bash
python slo_recon_hires.py                       # default: U=2, dense warp, RL deconv
python slo_recon_hires.py --U 1 --warp none     # ablation: no SR, global-rigid only
python slo_recon_hires.py --deconv none          # registered average without deconv
```

**Files**
- `slo_recon_hires.py` — raw-frame registration to iterative reference, quality
  gating, dense 2D warp, 2× supersampled quality-weighted shift-and-add,
  flat-field + two-axis destripe + Richardson-Lucy.
- `results/slo_recon_hires*.png`, `cache/slo_recon_hires_summary.npz`.
