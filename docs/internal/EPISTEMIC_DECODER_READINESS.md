# Epistemic Decoder Readiness

SOTA rate is not the objective. The objective is a trustworthy
epistemic-decoder experiment: ground-truth uncertainty, hard timing, calibrated
display, certified tracker precision, and an offline decoder analysis that
proves whether fine SLO features add predictive power beyond coarse gaze/pupil.

## Current Status

The repo has useful tracker infrastructure:

- `data.py`: SLO/raster loaders and calibration assets.
- `raster_attention.py`: raster strip tracking and SLO attention export.
- `results/gaze_study_verdict.md`: prior kHz raster verdict.
- `results/khz2d_methods.md`: current method comparisons.

This is enough for tracker development and offline analysis. It is not enough
for the epistemic-decoder claim until the experiment track below is complete.

## Minimum Readiness Spec

- Gaze: >=500 Hz floor, target about 1 kHz. Do not optimize for 15 kHz as the
  primary experimental path.
- Precision: <=1 arcmin RMS relative precision for any fine-band SLO advantage
  claim. <=2 arcmin can be a pilot-only tolerance.
- Pupil: >=30 Hz area/diameter with luminance regressors.
- Display: luminance calibrated, mean-luminance clampable,
  photodiode-verified frame timing.
- Sync: <=1 ms verified alignment among display, gaze, pupil, and task log.
- Stimuli: every trial emits ideal-observer labels for surprise, posterior
  entropy, and expected information gain.
- Analysis: compare coarse features vs coarse + fine SLO features under
  cross-validation.

## Experiment 1 Stimulus/Logging

`epistemic_task.py` generates the first real task substrate: a Gabor-orientation
change-point sequence with ideal-observer labels for every displayed frame.

Example:

```bash
python -m epistemic_task --out results/experiment1_demo --trials 4 --frames 240 --seed 0
```

Outputs:

- `experiment1_events.csv`: one row per displayed frame/event.
- `experiment1_manifest.json`: config, orientation grid, event columns, and
  summary statistics.

Required event columns include:

- `trial_id`, `trial_index`, `frame_in_trial`, `global_frame_id`,
  `display_frame_id`, `time_s`
- `orientation_deg`, `orientation_bin`, `latent_orientation_deg`, `change`,
  `hazard`
- `surprise_nats`, `posterior_entropy_nats`,
  `expected_information_gain_nats`, `information_gain_nats`
- `mean_luminance_cd_m2`, `luminance_cd_m2`, `gabor_contrast`

The ideal observer is a discrete Bayesian change-point observer over axial
orientation bins. At each frame it computes:

- Surprise: `-log p(observed_orientation | history)`.
- Posterior entropy: entropy of `p(latent_orientation | history, observation)`.
- Expected information gain: expected entropy reduction before the sample,
  under the observer's predictive distribution.

## Hardware Sync Capture Contract

Every real session needs one shared alignment file that can be validated against
`docs/schemas/sync_alignment.schema.json`. The capture must include:

- Photodiode pulse channel tied to display onset or a defined display patch.
- TTL/event channel where available.
- Tracker timestamps for gaze samples.
- Pupil timestamps for area/diameter samples.
- Display frame IDs matching `display_frame_id` in the task CSV.
- Task event log with `trial_id`, `frame_in_trial`, and stimulus labels.
- Alignment model with residual RMS, p99, and max absolute error in ms.

The session is not experiment-grade unless `verified_le_1ms` is true and the
reported max absolute alignment error is <=1 ms across display, gaze, pupil, and
task log.

## Tracker Certification Before Subjects

Do this before any subject-facing epistemic claim:

- Fixation or phantom precision with RMS in arcmin on both axes.
- Split-half repeatability on real data for the fine SLO channel.
- Fixed-time-lag displacement curves for the channel being used.
- No adjacent-sample speed metric as the primary proof of precision.
- Explicit pass/fail report for >=500 Hz floor, about 1 kHz target, and <=1
  arcmin fine-band claim threshold.

## Pilot Analysis

Pilot with a commercial 1 kHz tracker if available. Use SLO as a secondary
fine-band channel. The success criterion is not line-rate; it is that fine SLO
features add cross-validated predictive power beyond coarse gaze and pupil
features on the labeled uncertainty task.

The clean strategy is: 1 kHz, ground-truth uncertainty, hard sync, calibrated
display, offline decoder first. Line-rate PF/resolver work is now an ablation,
not the main path.
