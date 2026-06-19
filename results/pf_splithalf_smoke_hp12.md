# Split-half repeatability — the decisive 3' floor test

`RMS[(A-B)/sqrt2]` on disjoint line halves, interpolated to a common half-rate grid. `floor_hf` = full-run single-trace HF precision (25 ms high-pass) at operating point `b3_rp0.15`. All in arcmin (scale 0.4762'/px).

| subj | mode | band | floor_hf | split | ratio | split_all | full r·dot | A/B r·dot | A/B prec | n |
|------|------|------|---------:|------:|------:|----------:|-----------:|----------:|---------:|---:|
| Igor | evenodd | hf | 3.881 | 3.452 | 0.89 | 3.474 | 0.856 | 0.85/0.85 | 4.98/4.84 | 14542 |
| Igor | block | slow | 3.881 | 3.268 | 0.84 | 6.515 | 0.856 | 0.82/0.87 | 5.68/5.30 | 904 |

**Igor verdict:** MEASUREMENT-LIMITED: even/odd halves disagree at HF (split_hf=3.45' vs floor=3.88', ratio=0.89) -> the HF floor is per-line measurement noise, NOT real eye motion. -> Phase D-measurement (observation model). Frame-parity slow disagreement is also large (slow=3.27') -> atlas/decoder REGISTRATION error is a second contributor.
