# gaze-model

Kilohertz two-dimensional gaze reconstruction from a scanning ophthalmoscope.
A multimodal **particle filter** turns each fast scan line into a gaze estimate
by analysis-by-synthesis: every particle renders the line it would produce from
a recent frame, an oculomotor (saccadic main-sequence) prior propagates the
cloud, and a slow 2-D registration re-anchors it when appearance becomes
uninformative. This resolves the spatial aliasing that defeats single-hypothesis
(argmax / Kalman) trackers at high rate.

**Project page (paper writeup):** `docs/index.html` →
`https://strangecradles.github.io/gaze-model/` once Pages is enabled.

## Method components

| File | Role |
|------|------|
| `filter.py` | Multimodal particle filter (predict / weight / estimate / reseed) |
| `dynamics.py` | Interacting-multiple-model prior (pursuit OU + saccade main sequence) |
| `decoder.py` | Frozen differentiable line renderer (atlas → line) |
| `likelihood.py` | Physics appearance likelihood (aliased perp score) |
| `khz2d_methods.py` | The M0–M5 candidate methods and the benchmark harness |
| `losses.py`, `train.py` | Self-supervised losses + optional learned likelihood |

See `GOALS.md` and `PLAN.md` for the staged build (G1–G15) and `results/` for the
per-gate reports and figures.

## Reproduce the figures

```bash
python3 docs/make_figures.py        # regenerates docs/figures/fig_*.png
python3 -m http.server 8000 --directory docs
```

## Data note

Raw captures, the per-person atlas, caches, and result videos (several GB total)
are **not** tracked here (see `.gitignore`); the repository contains source, the
project page, and lightweight result summaries/figures only.

## Status

Research preprint. Real-data numbers are validated by self-consistency and
independent-tracker agreement (necessary, not sufficient); absolute-accuracy
validation against an artificial eye is future work — see the roadmap on the
project page.
