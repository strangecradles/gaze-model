# GOALS.md — Sequential `/goal` prompts for the 2D tracker build

Each block is the text for one Claude Code `/goal` (prepend `/goal`). Run in order; each assumes the previous merged with its tests green. `PLAN.md` (the NN build spec) is in the repo root.
**Hard-stop gates:** G5 (substrate) and G13 (decisive synthetic gate).

---

### G1 — Repo scaffold + data contracts
```
Set up the repo skeleton and a single data-loading module; read PLAN.md for context. Create data.py loading the imported assets and exposing each with documented shape, dtype, and physical units: (a) per-person atlas/enrollment map; (b) raw ~11 kHz line-scan stream; (c) metric along-shift series; (d) coarse-anchor perp series; (e) frame-rate full-frame-registration truth; (f) machine pupil-tracker (right_x/right_y) for kinematics. Add a Units dataclass; assert sample rates and shapes on load. Add tests/test_data.py asserting shape/rate/finite invariants for each asset. DoD: pytest tests/test_data.py passes and `python -m data --summary` prints a table of every asset's shape, rate, and units. Implement no modeling.
```

### G2 — Calibration (row↔arcmin)
```
Implement calib.py: derive the atlas-row↔arcmin scale from the along channel. Detect saccades on the 12 kHz along-shift series, fit the saccade main sequence (peak-velocity vs amplitude), and use the known sample clock to derive deg/row; cross-check against the frame-rate full-frame-registration truth scale. Expose rows_to_arcmin/arcmin_to_rows and a measured constant ALIAS_SPACING_ROWS (expect ~126 rows ~1deg, but measure). Add tests/test_calib.py asserting the two independent scale estimates agree within tolerance and the prior NaN path is gone. DoD: tests pass; `calib.py --report` prints deg/row, alias spacing in rows and arcmin, and the cross-check residual.
```

### G3 — Frozen physics decoder
```
Implement the frozen differentiable decoder in decoder.py: render(perp, along, atlas) -> line that samples the atlas at a 2D gaze via differentiable bilinear interpolation (grid_sample), returning the 1D line that gaze produces. No learned parameters; gradients must flow w.r.t. perp and along. Add tests/test_decoder.py: (a) a known (perp, along) renders a line matching a brute-force numpy reference within 1e-4; (b) autograd gradient matches finite-difference. DoD: tests pass.
```

### G4 — Noise + motion-blur model
```
Add noise.py: a measured-SNR noise model and a saccade motion-blur model for rendered lines. Noise is parameterized by the measured per-rate SNR curve (peak-corr vs rate; see PLAN.md). Blur integrates the atlas over gaze displacement during a line's integration time at given velocity. Expose apply_noise(line, rate) and apply_blur(atlas, perp, along, velocity, dt). Add tests/test_noise.py asserting output SNR matches the target per-rate value and that blur monotonically reduces high-spatial-frequency content with velocity. DoD: tests pass.
```

### G5 — Decoder validation on real held-out lines  *(HARD STOP)*
```
Validate the decoder against REAL data. Using frame-rate full-frame-registration truth to supply approximate (perp, along) for held-out real lines, render predicted lines and measure reconstruction error vs observed real lines. Write validate_decoder.py reporting the reconstruction-error distribution and its ratio to the measured noise floor. DoD: report median reconstruction error on held-out real lines; it must be within the noise floor. If it is not, STOP and flag the atlas/sampling mismatch — do not proceed. This gate proves the substrate before any filtering.
```

### G6 — Multimodal observation likelihood
```
Implement likelihood.py: perp_likelihood(line, atlas, along) -> score_over_rows, the physics atlas-match score across all candidate perp rows for a line (fixing along), producing the multimodal/aliased likelihood. Use the atlas-match score; learn nothing yet. Add tests/test_likelihood.py asserting: (a) for a synthetic line rendered at a known row, the true row is among the top peaks; (b) inter-peak spacing equals ALIAS_SPACING_ROWS from calib; (c) locked-peak width corresponds to sub-0.1deg. DoD: tests pass.
```

### G7 — Trajectory generator (label-bearing)
```
Implement traj_gen.py: generate known 2D gaze trajectories of fixation (drift + microsaccades), smooth pursuit, and saccade segments, with saccade kinematics drawn from the real main sequence measured in calib.py / the machine tracker. Expose sample_trajectory(duration, rate, seed). Add tests/test_traj_gen.py asserting the trajectories reproduce real oculomotor statistics: saccade main sequence, drift spectrum shape, microsaccade rate within plausible ranges. DoD: tests pass; `traj_gen.py --plot` shows a sample trajectory and its main-sequence scatter.
```

### G8 — Synthetic line-scan stream (rate-sweepable)
```
Implement synth_stream.py: given a trajectory from traj_gen, sample the atlas along it via decoder.py and apply noise.py to produce a synthetic 1D line-scan stream at a settable effective rate (subsample/average the raw line rate). Return paired (known_trajectory, line_stream, rate). Add tests/test_synth_stream.py asserting: per-rate SNR matches the measured curve; alias structure is present (likelihood on synthetic lines shows expected peak spacing); rate is settable from ~344 Hz to line rate. DoD: tests pass. This is the labeled substrate for the decisive validation.
```

### G9 — IMM dynamics model
```
Implement dynamics.py: an interacting-multiple-model prior with a pursuit mode (low acceleration, smooth velocity) and a saccade mode (ballistic, main-sequence peak-velocity-vs-amplitude), plus mode-transition probabilities. Expose predict(state, dt) returning propagated particle states and mode posteriors. Forbid any generic-smoothness fallback. Add tests/test_dynamics.py asserting a pure-prior rollout reproduces the saccade main sequence and plausible fixation drift. DoD: tests pass.
```

### G10 — Particle-filter core
```
Implement filter.py: a particle filter over state (perp, along, v_perp, v_along, mode) with MULTIMODAL belief. Predict uses dynamics.py; update fuses the multimodal perp likelihood (likelihood.py) with the trusted metric along-shift as a low-variance measurement; resample with effective-sample-size monitoring; output a per-step posterior distribution, not a point. No unimodal Gaussian collapse. Add tests/test_filter_fixation.py: on a synthetic stream WELL ABOVE 820 Hz, recover the known trajectory to sub-0.1deg through fixation segments. DoD: test passes; report arcmin RMS.
```

### G11 — Reacquisition / reseed
```
Add reacquisition to filter.py: when effective sample size collapses, reseed particles from the coarse-anchor absolute position (+-~43') plus the along measurement; the coarse anchor enters as an ABSOLUTE measurement, not a veto. This fixes the gate-runaway / infinite-coast failure. Add tests/test_reacquire.py: inject an artificial mid-stream lock loss and assert the filter reseeds and recovers within a bounded number of steps, with NO indefinite coast and persistence below a set cap. DoD: test passes.
```

### G12 — Through-saccade tracking
```
Verify the filter tracks through saccades. On synthetic streams at >=~1-2 kHz, run filter.py across segments containing saccades and measure residual gross-error and arcmin accuracy broken out by velocity (fixation vs saccade). Write eval_saccade.py reporting both, plus lock-rate vs velocity. DoD: through-saccade arcmin RMS is sub-0.1deg at >=~1-2 kHz and saccade-time gross-error does not exceed the fixation baseline by more than a small margin (report numbers). If saccade accuracy craters, determine whether the cause is lock loss during blur (lock-rate vs velocity) and flag it.
```

### G13 — Rate-sweep validation  *(DECISIVE GATE)*
```
Build rate_sweep.py: run the full filter on synthetic streams across effective rates frame -> ~820 Hz -> ~2 kHz -> line rate. At each rate report residual gross-error, arcmin accuracy, and persistence/run-length, broken out by velocity. Produce a 3-panel figure (accuracy vs rate; persistence vs rate; lock-rate vs rate). DoD: demonstrate whether (a) persistence collapses above ~820 Hz as predicted and (b) a high-rate window achieves sub-0.1deg. Write the verdict to results/rate_sweep_verdict.md. This is decisive: if it fails with perfect labels and perfect physics, the passive approach is dead — stop and report.
```

### G14 — Self-supervised losses (+ optional learned likelihood)
```
Implement losses.py and train.py for learned components, adding a learned calibrated observation likelihood ONLY if the physics score is the measured bottleneck from G13. Losses: L_recon (line reconstruction through the frozen decoder), L_dyn (off-manifold / main-sequence / band-limit penalty), L_couple (along<->perp saccade-direction consistency), optional L_anchor (slow component vs frame-rate truth). Train/validate on labeled synthetic so accuracy is checked directly. DoD: training is stable and does not degrade G13 synthetic accuracy; report whether the learned likelihood reduces residual gross at the operating rate.
```

### G15 — Real-data deployment + validation  *(GATE 2)*
```
Deploy the filter self-supervised on the REAL line-scan stream (losses recon + dyn + couple; no trajectory labels). Write eval_real.py validating by the only available real-data means: (a) slow component vs frame-rate full-frame-registration truth; (b) reproduction of oculomotor statistics (drift spectrum, microsaccade rate, main sequence); (c) alias-structure collapse with rate matching the synthetic prediction. Record results in results/real_validation.md, noting explicitly that self-consistency is necessary, not sufficient. DoD: the three checks are reported with numbers; flag clearly whether real performance matches the synthetic prediction or diverges.
```