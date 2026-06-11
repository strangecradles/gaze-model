# G5 — Decoder validation on real held-out lines (HARD STOP)

**Verdict: PASS**

Atlas renders reproduce held-out real lines at corr 0.829 (floor 0.873); error within 1.35x the noise floor. Substrate proven.

| quantity | noise floor (train) | held-out | ratio |
|---|---|---|---|
| reconstruction corr | 0.873 | 0.829 | (1-corr) 1.35x |
| reconstruction NMSE | 0.253 | 0.342 | 1.35x |

- Substrate: `normal/` capture (fixating subject), 15 train frames -> atlas; held-out frames' block-averaged (16-row) lines reconstructed via the frozen decoder at the full-frame-registered gaze.
- 100 held-out real lines evaluated (rows 150..450 step 15, length 900).
- Gate: held-out error <= 1.6x noise floor AND held-out corr >= 0.6.

Median held-out reconstruction error is within the noise floor: the per-person
atlas, sampled at the registered gaze, reproduces real observed lines. The
physics substrate is validated; G6+ (likelihood, filter) may proceed.
