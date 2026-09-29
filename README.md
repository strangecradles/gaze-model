# gaze-model

Kilohertz 2-D gaze from SD-SLO via multimodal particle filter (analysis-by-synthesis).

## Results (from project page)

- Synthetic @ 12 kHz: fixation/pursuit RMS 0.86′; persistence collapses above ~826 Hz aliasing gate
- Real test1: PF Prec x 1.76′ vs strip 4.39′ / SOTA strip 4.38′ @ line rate (see Pages tables)
- AOSLO public GT: 2-D RMS 0.055′ vs strip 0.123′ (simulated cone set; caveats on Pages)

Full writeup: https://strangecradles.github.io/gaze-model/

## Related

Strip-registration + flyback package: https://github.com/strangecradles/sdslo-gaze

## Method (one paragraph)

Particle filter keeps aliased perp belief multimodal; IMM main-sequence prior; frozen atlas decoder; slow 2-D anchor for reacquire.

## Status

Research preprint. Absolute artificial-eye GT still future work.
