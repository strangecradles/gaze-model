# Intra-frame atlas dewarp — verdict

**Question.** What fraction of the ~4.33′ per-line horizontal localization scatter
(`rdx_scatter = std(rdx − 10 ms smooth)`, Igor/pursuit_fov) is caused by uncorrected
**intra-frame (rolling-shutter) eye-motion distortion of the previous-frame atlas**?

**Answer (one line).** Essentially **none** — the hypothesis is **rejected (H0)**.
Dewarping the atlas does not reduce the scatter; it makes it *worse* on both subjects.
The scanner control is flat, so the axis assignment is correct. The named alternative —
single-frame reference noise — accounts for a **10–12 %** removable component, but that
reduction **fails the signal-preservation gate on Ashton3**, so it is not a clean win
either. ~**88–100 %** of the floor remains an unexplained structured systematic.

---

## Headline numbers

| Quantity | Igor | Ashton3 |
|---|---|---|
| Baseline `rdx_scatter` (Task 1) | **4.334′** (full mask) | 4.10′ |
| A/B dewarp ON vs OFF (common mask, Task 4) | 4.264 → 4.507′ | 4.093 → 4.603′ |
| **fraction_owned = (off−on)/off** | **−0.057** | **−0.125** |
| bootstrap 95 % CI (over frames) | [−0.068, −0.046] | [−0.138, −0.112] |

`fraction_owned` is **negative with a CI excluding zero on both subjects**: the dewarp
*adds* scatter. Decomposing the degradation with a sign-flip control (Igor):
OFF 4.217′, +sign (stabilize) 4.447′, −sign (anti-stabilize) 4.489′. Both signs are
worse by ~0.23–0.27′, dominated by **cubic-resample noise injected into the atlas**
(common to both signs). The physically-correct +sign is only 0.042′ better than −sign,
so the *real* intra-frame **drift**-distortion signal is ≈ 0.02′ ≈ **0.5 %** of 4.33′ —
negligible, and net-negative once resampling cost is paid.

**Why H1 fails mechanistically.** The metric residual is high-frequency (faster than the
10 ms smooth). Intra-frame *drift* is slow (a ramp across the 55 ms / 808-column frame)
and is therefore already removed by the 10 ms smooth — it never enters the residual, so
removing it from the atlas cannot lower the residual. The HF residual is owned by some
other, faster systematic.

## Scanner control (Task 2c) — FLAT ✓

Signed regression of residual on fast-axis (vertical/resonant) **position** `lam_v`:
**R² = 0.000** (Igor), position-bias peak-to-peak 0.61 px across the full vertical range.
The resonant scanner is exonerated; the 4.33′ is correctly assigned to the slow/galvo
horizontal axis. (The `|resid|`–`|lam_v|` amplitude coupling, R² = 0.096, is motion-
amplitude coupling — large motion → large scatter on both axes — which *supports* a
motion-driven systematic, not a scanner position artifact.)

Other regressions: (a) residual vs within-frame column R² = 0.003 pooled, with a weakly
consistent within-frame ramp (67 % of frames positive slope) — the slow component the
10 ms smooth removes; (b) per-frame scatter vs per-frame displacement **R² = 0.33** — the
floor scales with how much the eye moved, i.e. it is motion-driven measurement error.

## Reference-quality control (Task 5) — partial, fails anti-gaming on Ashton3

K-frame chain-averaged reference vs single previous frame (common mask):

| variant | Igor | Ashton3 |
|---|---|---|
| single frame | 4.316′ | 4.102′ |
| refavg3 | 3.796′ (**−12.0 %**) | 3.668′ (**−10.6 %**) |
| refavg5 | 4.063′ (−5.9 %) | — |

Averaging the reference lowers scatter (sqrt-K lower template noise), non-monotonic in K
(K=3 best; more frames add stale, differently-distorted content). **But** this is the
target of the signal gate, not a free lunch (next section).

## Signal-preservation gate (Task 6) — refavg3 PASSES on Igor, FAILS on Ashton3

| candidate | r-vs-dot Δ | microsaccades | even/odd coherence crossover | verdict |
|---|---|---|---|---|
| Igor refavg3 | +0.000 | 751 → 742 (−1.2 %) | 24 → **25 Hz** (up) | **PASS** |
| Igor dewarp | +0.002 | 751 → 664 | 24 → 22 Hz (down) | FAIL |
| Ashton3 refavg3 | +0.000 | 936 → 833 (−11 %) | 118 → **63 Hz** (down) | **FAIL** |
| Ashton3 dewarp | −0.001 | 936 → 691 | 118 → 82 Hz (down) | FAIL |

On Ashton3, whose real eye-motion signal is coherent out to ~118 Hz, averaging the
reference **halves the coherence crossover and removes 11 % of microsaccades** — i.e. it
buys part of its 10.6 % "improvement" by destroying real high-frequency motion. The gate
correctly flags this. On Igor (signal only coherent to ~24 Hz) refavg3 removes mostly
noise and passes. **Therefore reference averaging is not a robust, signal-safe reduction
across subjects**, and we do not attribute a clean fraction to reference noise.

(The microsaccade detector is coincidence-based across even/odd half-lines but still runs
~11 events/s ≫ physiological ~2/s, so it is noise-contaminated; the gate uses a ±5 %
tolerance and treats the **coherence crossover** as the robust signal-preservation test.)

## Exit-criteria checklist

- Tasks 1–7 ran and emitted artifacts: `dewarp_baseline.json`, `dewarp_regressions.json`,
  `dewarp_ab.json`, `dewarp_refquality.json`, `dewarp_signal_check.json`, this verdict,
  and `tests/test_dewarp_atlas.py` (7 passing).
- **Fraction owned by intra-frame distortion:** ≈ **0 %** (point estimate negative;
  95 % CI [−6.8 %, −4.6 %] Igor, [−13.8 %, −11.2 %] Ashton3 — the dewarp is strictly
  worse). The isolated drift-distortion signal is ~0.5 % of 4.33′.
- **Scanner control (2c) flat:** yes (R² = 0.000).
- **Signal gate (6):** the only candidate that *reduced* scatter (refavg3) passes on
  Igor but fails on Ashton3 → not a clean, cross-subject signal-safe win.
- **Residual unexplained floor after dewarp:** unchanged-to-worse; ≈ 4.3–4.6′. After the
  best *legitimate* intervention (refavg3, where it passes): ≈ 3.8′ on Igor (~88 % of
  baseline), no legitimate reduction on Ashton3.
- Implementation flag-gated (`dewarp_atlas=False`, `ref_frames=1` defaults); OFF reads the
  committed cache byte-identical; tests pass.

## Conclusion and next suspect

**H1 is rejected.** Intra-frame rolling-shutter distortion of the previous-frame atlas
owns ≈ 0 % of the 4.33′ horizontal residual: the residual is high-frequency, intra-frame
drift is slow and already removed by the 10 ms smooth, and naive atlas dewarp only injects
resample noise. The scanner is independently exonerated (control flat).

The floor is a **fast, motion-scaling, structured measurement systematic** (per-frame
displacement explains R² = 0.33 of it). Single-frame reference noise is a real but
**subject-dependent and signal-unsafe** ~10–12 % component, not the dominant cause.

**Named next suspects** (for a future loop):

1. **Intra-frame *microsaccade* distortion** (the HF part of intra-frame motion the chain
   ramp cannot capture) — would require a sub-frame, microsaccade-resolving gaze trace,
   tested against the same signal gate.
2. **Chain-anchor / inter-frame registration jitter** propagating into the per-line
   horizontal argmax (the per-frame displacement coupling, R² = 0.33, points here).
3. **Atlas content / NCC argmax bias** on the slow axis (sub-pixel parabolic-peak bias,
   contrast-dependent), independent of eye motion.
