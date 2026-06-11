# G14 — Self-supervised losses + learned calibrated likelihood

Trained on LABELED SYNTHETIC (synth_stream.make_synthetic). The frozen decoder renders the physics features; only the small calibration head is learned. Train seeds [0, 1, 2, 3, 4, 5] / held-out val seeds [6, 7, 8, 9] are DISJOINT.

Operating rate: 2000 Hz (the G13 fixation-sub-0.1-deg regime).

## Training stability (loss curve)

- epochs: 1; first 0.0000 -> last 0.0000 (decreased: False; NaN/inf: False)
- curve (sampled): 0.000

## Held-out physics-vs-learned perp localisation (argmax error)

Gross = fraction with |perp err| >= 0.5 deg; RMS in atlas rows (0.1 deg = 12.5 rows).

               |    gross |  RMS rows |  RMS arcmin
--------------------------------------------------------
FIXATION (n=6144)
  physics      |    0.000 |       0.3 |       0.14'
  learned      |    0.000 |       0.3 |       0.14'
SACCADE  (n=81)
  physics      |    0.222 |      78.5 |      37.78'
  learned      |    0.049 |      22.6 |      10.88'

## Does the learned likelihood reduce residual gross during saccades?

- saccade gross: physics 0.222 -> learned 0.049 (delta +0.173)
- saccade RMS:   physics 78.5 rows -> learned 22.6 rows (delta +55.9)
- fixation preserved: physics gross 0.000 / RMS 0.29 rows vs learned gross 0.000 / RMS 0.28 rows (sub-0.1 deg must survive)

**VERDICT: PARTIAL (learned reduces saccade gross/RMS but blur remains a hard physics limit — saccade still above 0.1 deg)**

## Robustness across disjoint held-out seed groups

Saccade lines are rare (~0.6%); each group is disjoint from the train seeds. Fixation stays sub-0.1 deg (12.5 rows) in every group.

val seeds        | sac n |     phys g/RMS |    learn g/RMS | fix learn RMS
--------------------------------------------------------------------------
[6, 7, 8, 9]     |    81 | 0.222/  78.5r | 0.049/  22.6r |        0.28r
[8, 9]           |    78 | 0.192/  65.1r | 0.051/  22.5r |        0.26r
[10, 11, 12, 13] |    51 | 0.314/  99.4r | 0.216/  72.7r |        0.25r
[14, 15, 16, 17] |    94 | 0.298/  83.5r | 0.096/  43.4r |        0.25r
