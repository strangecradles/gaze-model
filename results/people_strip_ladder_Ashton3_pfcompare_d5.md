# Ashton3 SLO Strip Ladder

Non-overlapping SLO-column strips registered to the previous frame.
`raw` is immediate top-1 NCC; `resolver` feeds top-K NCC peaks into the fixed-lag path resolver.

| method | S | rate Hz | valid | r_dot_x | prec_x | j30 | jump>=3 px | p99.9 speed px/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 64 | 223.2 | 0.953 | 0.923 | 4.816 | 0.716 | 0.090 | 13376.4 |
| resolver | 64 | 223.2 | 0.953 | 0.922 | 4.443 | 0.781 | 0.090 | 12487.5 |
| raw | 32 | 465.1 | 0.932 | 0.900 | 5.752 | 0.878 | 0.055 | 26537.3 |
| resolver | 32 | 465.1 | 0.932 | 0.894 | 5.418 | 0.808 | 0.048 | 13414.7 |
| raw | 16 | 930.2 | 0.930 | 0.904 | 6.053 | 1.037 | 0.057 | 52824.8 |
| resolver | 16 | 930.2 | 0.930 | 0.900 | 6.172 | 1.042 | 0.043 | 43109.7 |
| raw | 15 | 986.0 | 0.929 | 0.900 | 6.024 | 1.231 | 0.057 | 55906.1 |
| resolver | 15 | 986.0 | 0.929 | 0.895 | 5.993 | 1.074 | 0.039 | 40215.8 |
| raw | 8 | 1878.9 | 0.932 | 0.890 | 6.537 | 1.297 | 0.076 | 130742.1 |
| resolver | 8 | 1878.9 | 0.932 | 0.887 | 6.405 | 1.332 | 0.047 | 70164.9 |
| raw | 4 | 3757.8 | 0.936 | 0.894 | 6.699 | 1.092 | 0.086 | 257410.7 |
| resolver | 4 | 3757.8 | 0.936 | 0.889 | 6.721 | 1.025 | 0.047 | 134069.2 |
| raw | 2 | 7515.6 | 0.945 | 0.897 | 6.798 | 1.207 | 0.084 | 541126.2 |
| resolver | 2 | 7515.6 | 0.945 | 0.895 | 6.930 | 1.132 | 0.020 | 225776.8 |
| raw | 1 | 15031.3 | 0.956 | 0.889 | 6.849 | 1.227 | 0.100 | 1067221.2 |
| resolver | 1 | 15031.3 | 0.956 | 0.887 | 6.968 | 1.388 | 0.018 | 370233.5 |

Rates are `(frame_cols // S) * fps`; this capture has 808 columns, so `S=15/16` is the ~1 kHz baseline zone.
The report is a baseline characterization, not a replacement for PF caches.
