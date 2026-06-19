# Ashton3 SLO Strip Ladder

Non-overlapping SLO-column strips registered to the previous frame.
`raw` is immediate top-1 NCC; `resolver` feeds top-K NCC peaks into the fixed-lag path resolver.

| method | S | rate Hz | valid | r_dot_x | prec_x | j30 | jump>=3 px | p99.9 speed px/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 64 | 223.2 | 0.824 | 0.000 | 0.320 | 0.000 | 0.000 | 223.2 |
| resolver | 64 | 223.2 | 0.824 | 0.000 | 0.334 | 0.000 | 0.000 | 205.4 |
| raw | 32 | 465.1 | 0.778 | 0.000 | 0.426 | 0.000 | 0.000 | 930.2 |
| resolver | 32 | 465.1 | 0.778 | 0.000 | 0.374 | 0.000 | 0.000 | 749.4 |
| raw | 16 | 930.2 | 0.767 | 0.000 | 0.588 | 0.000 | 0.062 | 14627.6 |
| resolver | 16 | 930.2 | 0.767 | 0.000 | 0.534 | 0.000 | 0.011 | 9902.3 |
| raw | 15 | 986.0 | 0.776 | 0.000 | 0.547 | 0.000 | 0.081 | 11831.6 |
| resolver | 15 | 986.0 | 0.776 | 0.000 | 0.624 | 0.000 | 0.023 | 8560.2 |
| raw | 8 | 1878.9 | 0.761 | 0.000 | 1.067 | 0.000 | 0.122 | 46645.8 |
| resolver | 8 | 1878.9 | 0.761 | 0.000 | 0.993 | 0.000 | 0.082 | 39699.3 |
| raw | 4 | 3757.8 | 0.782 | 0.000 | 1.243 | 0.000 | 0.151 | 139039.4 |
| resolver | 4 | 3757.8 | 0.782 | 0.000 | 1.592 | 0.000 | 0.089 | 116829.2 |
| raw | 2 | 7515.6 | 0.814 | 0.000 | 1.634 | 0.000 | 0.190 | 375481.5 |
| resolver | 2 | 7515.6 | 0.814 | 0.000 | 1.725 | 0.000 | 0.041 | 161528.2 |
| raw | 1 | 15031.3 | 0.859 | 0.000 | 1.608 | 0.000 | 0.238 | 961596.3 |
| resolver | 1 | 15031.3 | 0.859 | 0.000 | 1.837 | 0.000 | 0.042 | 314394.4 |

Rates are `(frame_cols // S) * fps`; this capture has 808 columns, so `S=15/16` is the ~1 kHz baseline zone.
The report is a baseline characterization, not a replacement for PF caches.
