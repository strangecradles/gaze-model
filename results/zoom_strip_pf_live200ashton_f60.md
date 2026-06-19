# Zoom Synthetic-Reference Strip PF: live200ashton

Ground-truth-free diagnostics for strip observations matched to a coarse registered synthetic reference, then inferred with the IMM PF.
Assumed frame rate: 30 Hz; frames tracked: 60; reference frame index: 25; reference keep fraction: 0.842.

## Rate Scaling
| method | log-log p99.9 speed vs rate slope | n |
|---|---:|---:|
| raw_ref | 1.002 | 8 |
| pf | 0.796 | 8 |

Slope near 1 means fixed-size jumps are being differentiated by the sampling rate; flatter is better.

## Repeatability
| label | n | all RMS ' | HF25 RMS ' | slow50 RMS ' | slow corr |
|---|---:|---:|---:|---:|---:|
| raw_ref:S1_vs_S15 | 1996 | 7.418 | 7.097 | 1.197 | 0.940 |
| raw_ref:S1_vs_S16 | 1996 | 7.869 | 7.389 | 1.489 | 0.920 |
| raw_ref:S1_vs_S2 | 1997 | 3.857 | 3.743 | 0.442 | 0.993 |
| pf:S1_vs_S15 | 1996 | 18.487 | 7.570 | 15.041 | -0.091 |
| pf:S1_vs_S16 | 1996 | 17.471 | 8.104 | 14.337 | 0.389 |
| pf:S1_vs_S2 | 1997 | 16.494 | 8.287 | 12.691 | 0.318 |
| pf_split:evenodd:S15:f30 | 657 | 14.378 | 9.397 | 6.083 | 0.484 |
| pf_split:frameblock:S15:f30 | 616 | 18.522 | 5.527 | 16.504 | -0.518 |
| pf_split:evenodd:S1:f30 | 997 | 17.991 | 7.811 | 13.938 | -0.043 |
| pf_split:frameblock:S1:f30 | 934 | 16.556 | 6.304 | 12.686 | -0.117 |

## Evidence
| method | S | rate Hz | valid | max NCC med | ESS frac med | RMS vs immediate ' | jump>=3 | p99.9 speed px/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_ref | 64 | 300.2 | 0.793 | 0.138 | 0.650 | 0.000 | 0.150 | 29968.9 |
| pf | 64 | 300.2 | 0.793 | 0.138 | 0.650 | 23.117 | 0.142 | 17129.3 |
| raw_ref | 32 | 600.4 | 0.815 | 0.176 | 0.620 | 0.000 | 0.148 | 59988.8 |
| pf | 32 | 600.4 | 0.815 | 0.176 | 0.620 | 18.915 | 0.091 | 42567.4 |
| raw_ref | 16 | 1230.4 | 0.843 | 0.223 | 0.614 | 0.000 | 0.145 | 135086.3 |
| pf | 16 | 1230.4 | 0.843 | 0.223 | 0.614 | 16.601 | 0.095 | 98793.6 |
| raw_ref | 15 | 1320.3 | 0.841 | 0.203 | 0.612 | 0.000 | 0.155 | 211251.4 |
| pf | 15 | 1320.3 | 0.841 | 0.203 | 0.612 | 21.136 | 0.094 | 103466.5 |
| raw_ref | 8 | 2490.4 | 0.857 | 0.217 | 0.619 | 0.000 | 0.176 | 280363.4 |
| pf | 8 | 2490.4 | 0.857 | 0.217 | 0.619 | 19.089 | 0.079 | 216077.8 |
| raw_ref | 4 | 5010.2 | 0.866 | 0.290 | 0.604 | 0.000 | 0.174 | 572282.8 |
| pf | 4 | 5010.2 | 0.866 | 0.290 | 0.604 | 15.378 | 0.074 | 249075.8 |
| raw_ref | 2 | 10050.0 | 0.873 | 0.243 | 0.614 | 0.000 | 0.109 | 1075917.6 |
| pf | 2 | 10050.0 | 0.873 | 0.243 | 0.614 | 17.917 | 0.042 | 364953.2 |
| raw_ref | 1 | 20100.0 | 0.877 | 0.224 | 0.614 | 0.000 | 0.083 | 2132601.7 |
| pf | 1 | 20100.0 | 0.877 | 0.224 | 0.614 | 24.675 | 0.017 | 583876.8 |

No dot-correlation metric is reported here because these zoom TIFFs are not paired to the people-data pursuit target.
