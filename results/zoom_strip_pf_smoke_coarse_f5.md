# Zoom Synthetic-Reference Strip PF: live200ashton

Ground-truth-free diagnostics for strip observations matched to a coarse registered synthetic reference, then inferred with the IMM PF.
Assumed frame rate: 30 Hz; frames tracked: 5; reference frame index: 12; reference keep fraction: 0.550.

## Rate Scaling
| method | log-log p99.9 speed vs rate slope | n |
|---|---:|---:|
| raw_ref |  | 2 |
| pf |  | 2 |

Slope near 1 means fixed-size jumps are being differentiated by the sampling rate; flatter is better.

## Repeatability
| label | n | all RMS ' | HF25 RMS ' | slow50 RMS ' | slow corr |
|---|---:|---:|---:|---:|---:|
| raw_ref:S1_vs_S15 | 166 | 2.943 | 2.820 | 0.556 | 0.996 |
| pf:S1_vs_S15 | 166 | 19.655 | 3.957 | 17.032 | -0.942 |

## Evidence
| method | S | rate Hz | valid | max NCC med | ESS frac med | RMS vs immediate ' | jump>=3 | p99.9 speed px/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_ref | 15 | 1323.9 | 0.959 | 0.800 | 0.430 | 0.000 | 0.115 | 96071.0 |
| pf | 15 | 1323.9 | 0.959 | 0.800 | 0.430 | 9.469 | 0.019 | 29131.5 |
| raw_ref | 1 | 20100.0 | 0.976 | 0.842 | 0.489 | 0.000 | 0.096 | 1549349.6 |
| pf | 1 | 20100.0 | 0.976 | 0.842 | 0.489 | 23.910 | 0.008 | 437823.1 |

No dot-correlation metric is reported here because these zoom TIFFs are not paired to the people-data pursuit target.
