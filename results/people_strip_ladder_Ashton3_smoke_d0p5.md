# Ashton3 SLO Strip Ladder

Non-overlapping SLO-column strips registered to the previous frame.

| S | rate Hz | valid | r_dot_x | prec_x | j30 | jump>=3 px | p99.9 speed px/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 223.2 | 0.824 | 0.000 | 0.320 | 0.000 | 0.000 | 223.2 |
| 32 | 465.1 | 0.778 | 0.000 | 0.426 | 0.000 | 0.000 | 930.2 |
| 16 | 930.2 | 0.767 | 0.000 | 0.588 | 0.000 | 0.062 | 14627.6 |
| 15 | 986.0 | 0.776 | 0.000 | 0.547 | 0.000 | 0.081 | 11831.6 |
| 8 | 1878.9 | 0.761 | 0.000 | 1.067 | 0.000 | 0.122 | 46645.8 |
| 4 | 3757.8 | 0.782 | 0.000 | 1.243 | 0.000 | 0.151 | 139039.4 |
| 2 | 7515.6 | 0.814 | 0.000 | 1.634 | 0.000 | 0.190 | 375481.5 |
| 1 | 15031.3 | 0.859 | 0.000 | 1.608 | 0.000 | 0.238 | 961596.3 |

Rates are `(808 // S) * fps`; `S=15/16` is the ~1 kHz baseline zone.
The report is a baseline characterization, not a replacement for PF caches.
