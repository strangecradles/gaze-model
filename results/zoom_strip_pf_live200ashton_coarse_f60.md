# Zoom Synthetic-Reference Strip PF: live200ashton

Ground-truth-free diagnostics for strip observations matched to a coarse registered synthetic reference, then inferred with the IMM PF.
Assumed frame rate: 30 Hz; frames tracked: 60; reference frame index: 25; reference keep fraction: 0.842.

## Rate Scaling
| method | log-log p99.9 speed vs rate slope | n |
|---|---:|---:|
| raw_ref | 0.952 | 8 |
| pf | 0.856 | 8 |

Slope near 1 means fixed-size jumps are being differentiated by the sampling rate; flatter is better.

## Repeatability
| label | n | all RMS ' | HF25 RMS ' | slow50 RMS ' | slow corr |
|---|---:|---:|---:|---:|---:|
| raw_ref:S1_vs_S15 | 1999 | 2.361 | 2.348 | 0.112 | 1.000 |
| raw_ref:S1_vs_S16 | 1999 | 2.448 | 2.431 | 0.154 | 0.999 |
| raw_ref:S1_vs_S2 | 2000 | 1.665 | 1.627 | 0.185 | 0.999 |
| pf:S1_vs_S15 | 1999 | 17.217 | 9.732 | 10.406 | 0.036 |
| pf:S1_vs_S16 | 1999 | 16.569 | 9.791 | 9.037 | 0.091 |
| pf:S1_vs_S2 | 2000 | 17.322 | 8.585 | 10.136 | -0.214 |
| pf_split:evenodd:S15:f30 | 659 | 13.545 | 6.585 | 8.914 | 0.478 |
| pf_split:frameblock:S15:f30 | 616 | 12.099 | 5.419 | 7.984 | 0.638 |
| pf_split:evenodd:S1:f30 | 1000 | 18.921 | 7.313 | 15.557 | -0.761 |
| pf_split:frameblock:S1:f30 | 934 | 12.463 | 6.392 | 8.101 | 0.127 |

## Evidence
| method | S | rate Hz | valid | max NCC med | ESS frac med | RMS vs immediate ' | jump>=3 | p99.9 speed px/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_ref | 64 | 300.2 | 0.980 | 0.250 | 0.572 | 0.000 | 0.111 | 24397.7 |
| pf | 64 | 300.2 | 0.980 | 0.250 | 0.572 | 17.999 | 0.147 | 17313.6 |
| raw_ref | 32 | 600.4 | 0.985 | 0.293 | 0.611 | 0.000 | 0.060 | 47467.4 |
| pf | 32 | 600.4 | 0.985 | 0.293 | 0.611 | 15.656 | 0.112 | 41264.4 |
| raw_ref | 16 | 1230.4 | 0.986 | 0.235 | 0.635 | 0.000 | 0.048 | 95309.0 |
| pf | 16 | 1230.4 | 0.986 | 0.235 | 0.635 | 16.903 | 0.091 | 94706.0 |
| raw_ref | 15 | 1320.3 | 0.985 | 0.233 | 0.637 | 0.000 | 0.049 | 100931.2 |
| pf | 15 | 1320.3 | 0.985 | 0.233 | 0.637 | 17.632 | 0.102 | 130175.0 |
| raw_ref | 8 | 2490.4 | 0.988 | 0.352 | 0.630 | 0.000 | 0.062 | 143878.0 |
| pf | 8 | 2490.4 | 0.988 | 0.352 | 0.630 | 15.293 | 0.070 | 179282.1 |
| raw_ref | 4 | 5010.2 | 0.990 | 0.176 | 0.642 | 0.000 | 0.072 | 346346.6 |
| pf | 4 | 5010.2 | 0.990 | 0.176 | 0.642 | 20.279 | 0.068 | 309667.0 |
| raw_ref | 2 | 10050.0 | 0.990 | 0.308 | 0.614 | 0.000 | 0.053 | 667606.9 |
| pf | 2 | 10050.0 | 0.990 | 0.308 | 0.614 | 18.190 | 0.045 | 531300.4 |
| raw_ref | 1 | 20100.0 | 0.991 | 0.300 | 0.631 | 0.000 | 0.049 | 1409014.0 |
| pf | 1 | 20100.0 | 0.991 | 0.300 | 0.631 | 20.490 | 0.015 | 646960.9 |

No dot-correlation metric is reported here because these zoom TIFFs are not paired to the people-data pursuit target.
