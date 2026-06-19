# Strip-PF Reality Checks

Metrics are ground-truth-free checks of reproducibility and data support. `r_dot_x` is intentionally not used.

## Rate Scaling
| method | log-log p99.9 speed vs rate slope | n |
|---|---:|---:|
| raw | 1.063 | 8 |
| pf | 0.592 | 8 |

Slope near 1 means fixed-size measurement jumps are being differentiated; flatter is better.

## Repeatability
| label | n | all RMS ' | HF25 RMS ' | slow50 RMS ' | slow corr |
|---|---:|---:|---:|---:|---:|
| raw:S1_vs_S15 | 4925 | 2.749 | 2.375 | 1.216 | 0.999 |
| raw:S1_vs_S16 | 4646 | 3.008 | 2.501 | 1.463 | 0.998 |
| raw:S1_vs_S2 | 4997 | 2.033 | 1.523 | 1.313 | 1.000 |
| pf:S1_vs_S15 | 4925 | 4.167 | 2.292 | 2.305 | 0.993 |
| pf:S1_vs_S16 | 4646 | 4.216 | 2.313 | 2.451 | 0.991 |
| pf:S1_vs_S2 | 4997 | 4.220 | 2.299 | 2.610 | 0.993 |
| pf_split:evenodd:S15:d2 | 977 | 2.573 | 1.093 | 2.156 | 0.994 |
| pf_split:frameblock:S15:d2 | 926 | 6.539 | 1.524 | 6.171 | 0.930 |
| pf_split:evenodd:S1:d2 | 1987 | 2.444 | 2.113 | 0.884 | 0.999 |
| pf_split:frameblock:S1:d2 | 1881 | 7.042 | 2.143 | 6.204 | 0.923 |

## Evidence
| method | S | max NCC med | ESS frac med | multi hyp | RMS vs immediate ' | jump>=3 |
|---|---:|---:|---:|---:|---:|---:|
| pf | 64 | 0.586 | 0.585 | 1.000 | 4.878 | 0.037 |
| pf | 32 | 0.596 | 0.610 | 1.000 | 4.684 | 0.022 |
| pf | 16 | 0.608 | 0.616 | 1.000 | 3.954 | 0.012 |
| pf | 15 | 0.608 | 0.621 | 1.000 | 3.969 | 0.011 |
| pf | 8 | 0.610 | 0.630 | 1.000 | 4.333 | 0.009 |
| pf | 4 | 0.618 | 0.630 | 1.000 | 4.249 | 0.006 |
| pf | 2 | 0.624 | 0.630 | 1.000 | 4.691 | 0.006 |
| pf | 1 | 0.628 | 0.629 | 1.000 | 6.694 | 0.003 |
