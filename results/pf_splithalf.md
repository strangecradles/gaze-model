# Split-half repeatability — the decisive 3' floor test

`RMS[(A-B)/sqrt2]` on disjoint line halves, interpolated to a common half-rate grid. `floor_hf` = full-run single-trace HF precision (25 ms high-pass) at operating point `b3_rp0.15`. All in arcmin (scale 0.4762'/px).

| subj | mode | band | floor_hf | split | ratio | split_all | full r·dot | A/B r·dot | A/B prec | n |
|------|------|------|---------:|------:|------:|----------:|-----------:|----------:|---------:|---:|
| Ashton3 | evenodd | hf | 2.882 | 2.646 | 0.92 | 2.658 | 0.942 | 0.87/0.88 | 2.80/2.84 | 490953 |
| Ashton3 | block | slow | 2.882 | 3.451 | 1.20 | 6.908 | 0.942 | 0.88/0.87 | 3.62/3.43 | 31505 |

**Ashton3 verdict:** MEASUREMENT-LIMITED: even/odd halves disagree at HF (split_hf=2.65' vs floor=2.88', ratio=0.92) -> the HF floor is per-line measurement noise, NOT real eye motion. -> Phase D-measurement (observation model). Frame-parity slow disagreement is also large (slow=3.45') -> atlas/decoder REGISTRATION error is a second contributor.
