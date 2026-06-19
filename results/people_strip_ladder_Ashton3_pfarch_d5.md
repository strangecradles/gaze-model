# Ashton3 SLO Strip Ladder

Non-overlapping SLO-column strips registered to the previous frame.
`raw` is immediate top-1 NCC; `resolver` feeds top-K NCC peaks into the fixed-lag path resolver; `pf` runs the IMM particle filter on the strip NCC response surface, then resolves posterior clusters.

| method | S | rate Hz | valid | r_dot_x | prec_x | j30 | jump>=3 px | p99.9 speed px/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 64 | 223.2 | 0.953 | 0.923 | 4.816 | 0.716 | 0.090 | 13376.4 |
| pf | 64 | 223.2 | 0.953 | 0.895 | 2.838 | 0.301 | 0.037 | 10707.4 |
| raw | 32 | 465.1 | 0.932 | 0.900 | 5.752 | 0.878 | 0.055 | 26537.3 |
| pf | 32 | 465.1 | 0.932 | 0.861 | 3.044 | 0.625 | 0.022 | 14281.8 |
| raw | 16 | 930.2 | 0.930 | 0.904 | 6.053 | 1.037 | 0.057 | 52824.8 |
| pf | 16 | 930.2 | 0.930 | 0.886 | 3.770 | 0.967 | 0.012 | 30737.5 |
| raw | 15 | 986.0 | 0.929 | 0.900 | 6.024 | 1.231 | 0.057 | 55906.1 |
| pf | 15 | 986.0 | 0.929 | 0.876 | 3.734 | 0.725 | 0.011 | 34068.6 |
| raw | 8 | 1878.9 | 0.932 | 0.890 | 6.537 | 1.297 | 0.076 | 130742.1 |
| pf | 8 | 1878.9 | 0.932 | 0.870 | 4.092 | 0.998 | 0.009 | 54772.4 |
| raw | 4 | 3757.8 | 0.936 | 0.894 | 6.699 | 1.092 | 0.086 | 257410.7 |
| pf | 4 | 3757.8 | 0.936 | 0.880 | 4.563 | 0.971 | 0.006 | 87938.0 |
| raw | 2 | 7515.6 | 0.945 | 0.897 | 6.798 | 1.207 | 0.084 | 541126.2 |
| pf | 2 | 7515.6 | 0.945 | 0.882 | 4.557 | 0.907 | 0.006 | 98218.4 |
| raw | 1 | 15031.3 | 0.956 | 0.889 | 6.849 | 1.227 | 0.100 | 1067221.2 |
| pf | 1 | 15031.3 | 0.956 | 0.846 | 4.620 | 0.773 | 0.003 | 103589.8 |

Rates are `(frame_cols // S) * fps`; this capture has 808 columns, so `S=15/16` is the ~1 kHz baseline zone.
The report is a baseline characterization, not a replacement for PF caches.
