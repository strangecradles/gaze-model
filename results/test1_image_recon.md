# test1 SLO image reconstruction — naïve vs Azimipour+SOTA-strip vs Azimipour+OUR-DPF, on OUR OWN raster

**Question.** On our own real `test1` pursuit raster, reconstruct an SLO image
three ways that differ *only* in the eye-motion trace, and ask: does our
particle-filter (DPF) tracking reconstruct a **better** image than the SOTA
composite-reference strip method — and do either beat doing nothing about
intra-frame motion?

**Answer (headline, honest).**
On our real data there is **no meaningful, trustworthy image-quality difference
between the two trackers, and neither robustly beats a well-registered chain-only
average.** Visually the three reconstructions are near-identical (same optic
disc, same vasculature, cleanly registered). The strip arm shows a large *raw*
gradient-sharpness/structure number, but it **does not survive a paired
sub-composite test** (strip vs naïve sharpness p = 0.88) and is concentrated in
the **cross-column gradient** — the streak signature of the strip's ~3× noisier
per-column trace (its line-rate precision is 4.4′ vs our 1.6′,
`results/sota_comparison.md`), visible as extra speckle "crunch" in the strip
zoom. By the trustworthy paired metrics (structure-band power, contrast,
along-column detail) **our DPF arm is, if anything, marginally cleaner than the
strip arm**, but the effects are small and on a no-ground-truth proxy, so we do
**not** claim a real-image win. The reason all three converge: in this
fixation-dominated window the within-frame eye motion is sub-pixel to ~1 px
(median residual 0.77 px x, 0.63 px y), too small for intra-frame correction to
matter on noisy real frames. This matches the prior AO-SLO real-frame finding
("both ≈, both far better than naïve only when intra-frame motion is large").

---

## Frame-source confirmation

`calibration/video_playback_test1_20260605_100051_SLO_0.mp4` is **the same video**
the khz2d harness uses for test1: `data._slo_path("test1")` resolves to exactly
this path (`data.CAPTURES["test1"]["stem"] + "_SLO_0.mp4"`), and
`khz2d._read_frames()` / `khz2d.chain()` read it. So the cached eye traces align
**line-for-line** with the frames, and I used the khz2d frame pipeline directly
(no re-indexing needed). Frames: **1025 × (1000 rows fast/vertical, 808 cols
slow/horizontal)** at **14.633 fps**; chain anchor valid on 72% of frames.

## Geometry (adapted from `aoslo_image_recon.py`)

A test1 "line" (one ~85 µs fast sweep) is a **COLUMN** (axis 1 is the slow scan;
axis 0 is the fast scan), the **opposite** axis mapping from the AOSLO frames in
`aoslo_image_recon.py` (where a line is a row). The Azimipour dewarp (eqs 9–10)
therefore applies a **per-column** offset:

```
corrected[r, c] = frame( row = r + s·ry[c], col = c + s·rx[c] )
```

with `(rx[c], ry[c])` the per-column intra-frame residual and `s` a single global
sign chosen once by composite sharpness (came out −1; same for all arms). I
reused the AOSLO dewarp/averaging/metrics kernel and adapted only the axis
(offset indexed by column instead of row). A synthetic round-trip (warp a clean
texture by a known per-column trace, then dewarp by its negative) recovers the
interior to rms/std ≈ 0.11, confirming the column-axis kernel and sign.

## What is held identical across the three arms

Everything except the trace: per-frame **integer chain anchor** (the shared 20-Hz
absolute fix that carries the large inter-frame drift), the subpixel + intra
residual dewarp (cubic `map_coordinates`, order 3), the weighted register-and-
average, the crop, and the metrics. The three arms differ **only** in the
per-column residual that is added on top of the shared chain anchor:

- **naïve** — chain only (residual = 0): per-frame rigid registration, *no*
  intra-frame correction. The "do nothing about within-frame eye motion" baseline.
  (A raw unregistered average would be far worse — the eye drifts >250 px in this
  window — so chain-only is the strong, fair baseline that isolates the
  intra-frame benefit.)
- **strip** — `sota_strip.sota_roorda` composite-reference strip trace,
  per-column (S = 1), validity-matched at NCC threshold 0.35 (→ 61% valid,
  matched to DPF; the exact `sota_s1_d20` cache behind `results/sota_comparison.md`).
- **dpf** — our best config `m4_dpf` `learned_n1000_ess0.7_nw3`, per-column
  (block = 1), 60% valid (`results/real_eye_optimization.md`).

Each trace `(t, x_px, y_px)` (khz2d chain pixel coords) is interpolated onto the
808 per-column line times of every frame, identically for both arms.

## Validity / FOV handling

test1 has the right-eye temporal FOV dropout. Every contributed column is weighted
by a **shared, method-independent** content-validity mask (`khz2d.fov_mask`: line
ok & horizontal NCC > 0.30 & contrast floor) — in this window 92.7% of columns are
in-FOV. Because the weight and the integer placement are identical across arms,
the per-pixel coverage map is **byte-identical** across arms; only the pixel
*values* differ by the trace. Masked/invalid columns get zero weight in all three
arms, so the FOV dropout cannot bias the comparison.

## Compute / scope (stated)

The full 70 s drifts **more than a frame** (chain x spans 815 px, y 497 px), so an
all-1024-frame mosaic overlaps in only a tiny patch. I reconstructed on the
**0–20 s window** (182 valid frames, the validated line-rate window of
`results/sota_comparison.md`) using both methods' **per-column native** caches, so
there is no interpolation-density mismatch — the cleanest possible "only the trace
differs" comparison. Composite crop after coverage gating: 968 × 720 px.

---

## Results — full-window single composite (182 frames)

Reference-free metrics on the common crop. `sharp` = gradient energy; `sharp_x`
= cross-column gradient (inflated by per-column trace noise → vertical streaks);
`sharp_y` = along-column gradient; `struct` = mean radial power in the
0.05–0.25 cyc/px structure band; `HF` = high-frequency power fraction.

| arm | sharp | sharp_x (cross-col) | sharp_y (along-col) | contrast | struct | HF |
|---|---|---|---|---|---|---|
| naïve (chain-only) | 0.964 | 0.492 | 0.472 | 0.2471 | 3.17e6 | 0.0031 |
| Azimipour + SOTA strip | **1.449** | 0.748 | 0.701 | 0.2468 | **5.36e6** | 0.0035 |
| Azimipour + OUR DPF | 0.938 | 0.483 | 0.454 | 0.2467 | 3.07e6 | 0.0030 |

Taken at face value the strip arm looks ~50% sharper. **But these single-composite
numbers are not trustworthy for ranking** — they contradict the paired test below
and carry the streak-artifact signature (strip's edge is large in the
*cross-column* gradient, and the strip zoom is visibly speckle-crunchy rather than
showing more real vessels).

## Results — paired sub-composite effect sizes (the trustworthy test)

The window is split into **8 disjoint frame groups**; for each group a composite
is built per arm and each metric measured, giving a paired comparison (the same
protocol established as trustworthy in `results/aoslo_image_quality.md`). Mean Δ,
Cohen's d_z and paired-t p over the 8 groups:

| metric | DPF − strip | strip − naïve | DPF − naïve |
|---|---|---|---|
| sharp | −0.15 (d_z −0.48, p=0.22) | **+0.02 (d_z +0.05, p=0.88)** | −0.13 (d_z −1.21, p=0.011) |
| sharp_x (cross-col) | −0.43 (d_z −1.43, p=0.005) | **+0.36 (d_z +1.16, p=0.013)** | −0.07 (p=0.006) |
| sharp_y (along-col) | **+0.28 (d_z +1.86, p=0.001)** | **−0.34 (d_z −2.44, p=2e-4)** | −0.07 (p=0.028) |
| struct | **+5.2e6 (d_z +1.53, p=0.004)** | −5.4e6 (d_z −1.64, p=0.002) | −0.18e6 (p=0.59) |
| contrast | **+3.7e-3 (d_z +1.94, p=9e-4)** | −4.4e-3 (p=4e-4) | −0.7e-3 (p=0.035) |
| HF | −2.8e-3 (p=0.044) | +3.6e-3 (p=0.014) | +0.8e-3 (p=0.12) |

**Reading the numbers (honest).**
- **The strip arm's full-window sharpness/structure edge evaporates under the
  paired test:** strip vs naïve raw sharpness is a dead tie (p = 0.88), and strip
  is actually **lower** than naïve in along-column detail (sharp_y, p = 2e-4) and
  in structure-band power (p = 0.002). Its only consistent gains over naïve are in
  **cross-column gradient** (sharp_x, p = 0.013) and HF fraction (p = 0.014) — both
  the expected fingerprints of a noisy per-column trace injecting vertical streaks,
  not retinal detail.
- **DPF vs strip:** raw sharpness tied (p = 0.22); DPF has **less cross-column
  streaking** (sharp_x lower), **more along-column detail** (sharp_y higher,
  p = 0.001), **higher structure-band power** (p = 0.004) and **higher contrast**
  (p = 9e-4). So on the metrics that track real structure rather than streak
  noise, the DPF arm is **marginally cleaner** than the strip arm. Strip "wins"
  only HF fraction, consistent with its extra speckle.
- **Either tracker vs naïve:** DPF is statistically a wash with chain-only naïve
  (struct p = 0.59; a small negative on raw sharpness). Neither tracker delivers a
  robust improvement over chain-only registration, because the intra-frame motion
  to correct here is sub-pixel-to-~1 px.

See `results/test1_image_recon.png`: full composites (near-identical), zoom insets
(strip is crunchier, DPF ≈ naïve clean), metric bars, and the cross- vs
along-column streak-guard panel.

---

## Verdict (honest)

- **Does our DPF tracking reconstruct a meaningfully better test1 image than the
  strip method?** **No — but it is not worse, and is marginally cleaner.** Raw
  gradient sharpness is tied in the trustworthy paired test; on structure-band
  power, contrast, and along-column detail the DPF arm edges out the strip arm,
  while the strip arm's apparent raw-sharpness lead is a **streak artifact** of its
  ~3× noisier per-column trace (it doesn't survive the paired test and lives in the
  cross-column gradient). These DPF advantages are small and on a no-ground-truth
  proxy, so we do **not** claim a real-image win — we claim **parity, with our
  trace producing fewer noise streaks**.
- **Do either beat the naïve (chain-only) baseline?** **Not meaningfully**, on this
  window. Chain-only registration already aligns the frames well, and the residual
  within-frame motion is too small (median <1 px) for the Azimipour intra-frame
  dewarp to add trustworthy detail on real, speckle-limited frames.
- **Consistency with prior work.** This mirrors `results/aoslo_image_quality.md`:
  on *real* frames the trackers are essentially tied and the noisier estimator
  shows up as extra texture, not extra fidelity — the clear DPF win there was only
  on the *noiseless simulated* set where per-row/per-column sub-pixel accuracy
  translates directly into reconstruction fidelity.

## Caveats / blockers (none favors either arm)

- **No ground truth.** test1 is real; all metrics are reference-free proxies
  (gradient energy, contrast, radial structure power, cross/along-column gradient).
  Direction is reported with effect sizes, not over-claimed.
- **Single-composite metrics are unreliable here** (they contradict the paired
  test and carry the streak signature); the paired sub-composite effect sizes are
  the honest measure.
- **Window, not full recording** (0–20 s, 182 frames) because the full 70 s drifts
  >1 frame and the all-frames mosaic overlap is tiny. The window is the validated
  line-rate window of `results/sota_comparison.md`; both arms use per-column native
  traces in it.
- **Small intra-frame motion** in this fixation-dominated window is *why* the
  correction is marginal; a higher-motion segment would separate the arms more, but
  also raise noise — not pursued here to avoid cherry-picking.
- **Validity-matched SOTA threshold** (0.35 → 61%, matched to DPF 60%) as in
  `sota_comparison.md`; the shared content-FOV weighting is identical across arms.

## Reproduce

```bash
# prerequisites (already cached): khz2d.chain(), khz2d_lines.npz,
#   khz2d_sota_s1_d20.npz, khz2d_m4_dpf_11823_learned_n1000_ess0.7_nw3_d20.npz
python test1_image_recon.py        # writes results/test1_image_recon.png + cache summary
```

**Files**
- `test1_image_recon.py` — column-axis Azimipour dewarp, 3-arm weighted
  register-and-average, reference-free metrics + streak guard, paired effect sizes,
  figure.
- `results/test1_image_recon.png` — naïve | SOTA-strip | OUR-DPF composites +
  zoom insets + metric bars + cross/along-column streak-guard panel.
- `cache/test1_image_recon_summary.npz` — composites, metrics, effect sizes.
