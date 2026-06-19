# SDSLO single-line tracker upgrade

## What changed

This pass reframes the high-rate failure as SDSLO single-line appearance
ambiguity, not cone/mosaic-resolution tracking. The clean synthetic generator is
unchanged by default, but `synth_stream` now has explicit SDSLO stress presets:
short context, additive noise, structured residuals, optical blur mismatch, and
along-channel uncertainty.

The particle filter now has defaults-off SDSLO ambiguity controls:

- quality-scaled along likelihoods, so low-quality along measurements widen the
  Gaussian instead of forcing a bad lock;
- top-K particle-cluster hypotheses during ambiguous lines;
- mode-preserving resampling when the posterior is genuinely split;
- a fixed-lag top-K resolver in `filter.run`, defaulting to 1 ms when enabled.

The learned-likelihood training path can now build mixed clean/stressed datasets
and reports stress cases separately instead of pooling away SDSLO failures.

## Synthetic stress result

Focused test case: `combo_sdslo`, 2 kHz, 0.45 s, seeds 0 and 4, N=180. The
baseline PF is deliberately given the stressed along measurement; the upgraded PF
uses low along quality plus multi-hypothesis tracking.

| run | fixation RMS | gross-error rate | max gross persistence |
|---|---:|---:|---:|
| baseline PF | 42.68' | 0.382 | 109.75 ms |
| SDSLO upgrade | 12.71' | 0.065 | 26.50 ms |

Clean pure-fixation synthetic remains byte-identical with `stress="clean"` and
the upgraded PF preserves the clean fixation RMS in the targeted test.

## Ashton3 5 s real diagnostic

Matched 5 s caches, no animation/display smoothing:

| cache | r-vs-dot x | prec_x | j30 | valid |
|---|---:|---:|---:|---:|
| `m4_dpf_physics_d5` | 0.90 | 3.94' | 0.87' | 97.8% |
| `m4_dpf_physics_sdslo_d5` | 0.90 | 5.23' | 0.90' | 98.4% |

Raw line-rate step diagnostics:

| cache | p99 step | p99.9 step | max step | p99.9 speed | max speed | >=3 px jumps |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 4.51' | 26.53' | 48.45' | 6646.6 deg/s | 12137.6 deg/s | 0.0258 |
| SDSLO upgrade | 3.82' | 20.84' | 50.16' | 5221.0 deg/s | 12566.5 deg/s | 0.0236 |

Honest read: the opt-in PF reduces high-percentile raw jumps slightly, but this
configuration does **not** yet solve raw real-data line-rate velocity
implausibility and is not a replacement for the display oculomotor smoother. The
real-data rollout should stay opt-in while the along-quality calibration is
tuned on longer captures.

## Verification

- `pytest tests/test_sdslo_stress_tracker.py tests/test_synth_stream.py tests/test_filter_fixation.py tests/test_train.py tests/test_jitter_ablation.py tests/test_mosaic_reject.py tests/test_mosaic_prior.py tests/test_oculo_smooth_alias.py -q`
- Result: 33 passed.

