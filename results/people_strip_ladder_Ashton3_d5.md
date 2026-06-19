# Ashton3 SLO Strip Ladder

Non-overlapping SLO-column strips registered to the previous frame.

| S | rate Hz | valid | r_dot_x | prec_x | j30 | jump>=3 px | p99.9 speed px/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 223.2 | 0.953 | 0.923 | 4.816 | 0.716 | 0.090 | 13376.4 |
| 32 | 465.1 | 0.932 | 0.900 | 5.752 | 0.878 | 0.055 | 26537.3 |
| 16 | 930.2 | 0.930 | 0.904 | 6.053 | 1.037 | 0.057 | 52824.8 |
| 15 | 986.0 | 0.929 | 0.900 | 6.024 | 1.231 | 0.057 | 55906.1 |
| 8 | 1878.9 | 0.932 | 0.890 | 6.537 | 1.297 | 0.076 | 130742.1 |
| 4 | 3757.8 | 0.936 | 0.894 | 6.699 | 1.092 | 0.086 | 257410.7 |
| 2 | 7515.6 | 0.945 | 0.897 | 6.798 | 1.207 | 0.084 | 541126.2 |
| 1 | 15031.3 | 0.956 | 0.889 | 6.849 | 1.227 | 0.100 | 1067221.2 |

Rates are `(frame_cols // S) * fps`; this capture has 808 columns, so `S=15/16` is the ~1 kHz baseline zone.
The report is a baseline characterization, not a replacement for PF caches.
