# people_data_results jitter fix

## Diagnosis

The MP4 jitter came from rendering the raw line-rate PF trace. The recent PF work was right:
the IMM prior is not the main failure. Two measured effects dominate:

- PF gain noise: sharp `BETA=20` weights plus roughening/resampling inject line-rate jitter.
- Mosaic alias flips: the horizontal/slow-axis localizer can jump to a neighboring
  photoreceptor-mosaic peak. These jumps are discontinuous, but some persist long enough to look
  plausible after display sampling.

The canonical videos were generated with `style=raw`, so they exposed those artifacts directly.

## Fix

`oculo_smooth.oculomotor_trajectory()` now repairs only short jump-return alias excursions before
the event-preserving display smoother. Smooth ramps are left alone, and caller-supplied saccade
intervals are exempt. Raw PF caches are unchanged.

`people_data_fov_run.py --anim` now renders the de-aliased oculomotor style by default. Raw
diagnostic renders use `_raw.mp4`.

## Cached-trace metrics

`j30` is median absolute second difference after 30 fps sampling, in arcmin. `big` is the fraction
of valid adjacent line pairs with a >=3 px horizontal jump before display smoothing.

| subject | raw j30 | previous oculomotor j30 | fixed j30 | raw big | repaired big |
|---|---:|---:|---:|---:|---:|
| Ashton3 | 1.40 | 1.07 | 1.06 | 0.070 | 0.012 |
| Chong | 1.59 | 1.24 | 1.19 | 0.096 | 0.016 |
| Igor | 1.75 | 1.32 | 1.15 | 0.140 | 0.022 |
| John | 1.09 | 0.85 | 0.82 | 0.028 | 0.005 |
| Kathy | 1.53 | 1.17 | 1.17 | 0.064 | 0.009 |
| Pavel | 1.51 | 1.11 | 1.08 | 0.110 | 0.015 |
| Zohre | 2.10 | 1.63 | 1.63 | 0.156 | 0.037 |

All canonical `people_data_results/gaze2d_*_m4_dpf.mp4` files were regenerated from the fixed
display path. The previous raw renders were preserved as `*_raw.mp4`.

