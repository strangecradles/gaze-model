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
| raw_ref:S1_vs_S15 | 164 | 4.843 | 4.727 | 0.573 | 0.998 |
| pf:S1_vs_S15 | 164 | 19.335 | 3.766 | 17.120 | -0.879 |

## Evidence
| method | S | rate Hz | valid | max NCC med | ESS frac med | RMS vs immediate ' | jump>=3 | p99.9 speed px/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_ref | 15 | 1323.9 | 0.877 | 0.837 | 0.407 | 0.000 | 0.273 | 107719.6 |
| pf | 15 | 1323.9 | 0.877 | 0.837 | 0.407 | 13.241 | 0.070 | 74575.7 |
| raw_ref | 1 | 20100.0 | 0.923 | 0.122 | 0.534 | 0.000 | 0.124 | 1825428.4 |
| pf | 1 | 20100.0 | 0.923 | 0.122 | 0.534 | 33.238 | 0.005 | 350379.7 |

No dot-correlation metric is reported here because these zoom TIFFs are not paired to the people-data pursuit target.
