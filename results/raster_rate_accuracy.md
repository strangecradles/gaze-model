# test1 SLO raster — 2D gaze rate/accuracy frontier

Method: incremental strip registration (TSLO-style). Scale (per-phase vs 32.5 Hz tracker): along 0.584'/px (r=+0.62), perp 0.403'/px (r=+0.65).

| S | rate (Hz) | in-FOV % | r_dot Hx | r_dot Vy | r_dot Cx | r_dot Cy | r_dot Lx | r_dot Ly | NF perp (') | NF along (') |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 366 | 71 | +0.96 | +0.92 | +0.96 | +0.87 | +0.93 | +0.68 | 0.84 | 3.18 |
| 16 | 732 | 72 | +0.95 | +0.91 | +0.95 | +0.85 | +0.93 | +0.66 | 1.12 | 4.46 |
| 8 | 1478 | 73 | +0.95 | +0.92 | +0.96 | +0.85 | +0.93 | +0.65 | 1.35 | 5.07 |
| 4 | 2956 | 73 | +0.95 | +0.92 | +0.95 | +0.85 | +0.92 | +0.64 | 1.53 | 5.41 |
| 2 | 5912 | 74 | +0.95 | +0.92 | +0.95 | +0.85 | +0.91 | +0.64 | 1.75 | 5.76 |

r_dot = Pearson vs pursuit dot (target); H/V/C/L = H_sine/V_sine/circle/lissajous; x = horizontal (slow axis), y = vertical (fast axis). NF = truth-free >40 Hz noise floor.