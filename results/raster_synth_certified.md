# Certified strip-tracking accuracy vs rate (labeled synthetic)

Synthetic raster rendered from a clean test1 retina + known 2D gaze at the real per-column timing with measured per-rate noise; recovered by strip registration against the perfect reference. px->arcmin = 0.403'/px perp, 0.584'/px along (measured vs the 32.5 Hz tracker).

| S | rate (Hz) | lock % | FIX perp (') | FIX along (') | SAC perp (') | SAC along (') |
|---|---|---|---|---|---|---|
| 32 | 366 | 100 | 1.39 | 3.10 | 1.52 | 2.88 |
| 16 | 732 | 100 | 1.69 | 3.28 | 1.28 | 2.04 |
| 8 | 1478 | 100 | 1.79 | 3.59 | 1.32 | 2.44 |
| 4 | 2956 | 100 | 1.90 | 3.73 | 1.39 | 2.51 |
| 2 | 5912 | 100 | 1.97 | 3.81 | 1.36 | 2.42 |
| 1 | 11823 | 100 | 2.06 | 3.92 | 1.38 | 2.47 |

FIX = fixation/pursuit, SAC = through-saccade. perp = vertical (fast axis), along = horizontal (slow axis). RMS error vs the known label.