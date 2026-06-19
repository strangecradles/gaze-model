# Step 0 speckle recalibration (critique caveat 4)

The original Step 0 ablation used an **uncontrolled** per-line noise amplitude
(`OBS_NOISE_SIG=0.15`). Calibrating it to the real measured localization scatter shows
the original "0.061 rows ≈ 0.03′ floor" understated the true per-line noise by ~100×,
and — more importantly — that the real noise is **not white speckle at all**.

## Real single-line localization scatter (Igor)

`rdx` (per-line horizontal NCC argmax) minus a 10 ms-smoothed track:

**4.33′ = 9.10 raster px = 9.00 atlas rows.**

## Synthetic single-line argmax std vs injected white-noise amplitude

Same observation model as the filter (`decoder.render` + `filter._fine` + NCC argmax):

| obs_sig | argmax std (rows) | argmax std (′) | regime |
|--:|--:|--:|---|
| 0.05 | 0.081 | 0.039 | locked |
| 0.10 | 0.083 | 0.040 | locked |
| 0.15 | 0.090 | 0.043 | locked (original ablation amplitude) |
| 0.25 | 0.105 | 0.051 | locked |
| 0.40 | 0.136 | 0.066 | locked |
| 0.60 | 1.231 | 0.593 | **cliff** — argmax starts jumping alias peaks |
| 0.90 | 2.266 | 1.091 | alias-confused |
| 1.30 | 3.954 | 1.903 | alias-confused |
| 2.00 | 5.362 | 2.581 | alias chaos |

To reach the real 4.33′ single-line scatter with white speckle requires `obs_sig ≳ 2.0`,
i.e. the localizer is already in **alias chaos**, not stable localization.

## Filter-output ablation at the two regimes (med|Δ²|, HF step floor)

| config | obs_sig=0.4 (realistic, locked) | obs_sig=2.0 (matches real amp) |
|--------|--------------------------------:|-------------------------------:|
| A default (β20, rp0.5) | 0.082 rows (0.039′) | 156 rows (chaos) |
| C no roughen           | 0.0003 rows         | 166 rows (chaos) |
| D lock-gated           | 0.0011 rows         | 130 rows (chaos) |
| E β5 rp0.08            | 0.0031 rows (0.001′)| 141 rows (chaos) |

(`std` of the full trace at obs_sig=0.4 is inflated to ~7–46 rows by rare alias jumps;
the median second-difference `med|Δ²|` shown above is the HF step-jitter floor and stays
≈0.001–0.04′.)

## Conclusion

White Gaussian per-line noise is **conclusively excluded** as the ~3′ source, for two
independent reasons:

1. **At realistic amplitude** (single-line std ≤ ~0.07′), the filter's HF step floor is
   ~0.001–0.04′ — milliarcminute-to-0.04′ scale, ~100× below the real 3′.
2. **At the amplitude needed to reproduce the real 4.3′ scatter**, the NCC localizer
   disintegrates into alias chaos (std ~110 rows), which is nothing like the stable ~3′
   track actually observed.

The real 4.3′ per-line localization scatter is therefore **structured measurement error**
(atlas-template / per-frame registration mismatch), not additive white speckle. This
corroborates the split-half verdict: the floor is measurement-limited, with a large
slowly-varying registration component.
