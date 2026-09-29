# 2D Gaze Tracker — Neural / Differentiable-Filter Build Spec

## Objective
Learn the full 2D gaze trajectory `(perp(t), along(t))` from the 1D line-scan stream. This is a **state-estimation problem with a multimodal observation model and band-limited dynamics** — build a differentiable particle filter (MCL), not a sequence-to-sequence regressor.

## What has worked (the inputs and priors the model builds on)
- **Along axis — solved.** Sub-pixel metric registration (cross-correlation shift), 12 kHz, captures real eye motion (r ≈ 0.60 vs saccade-following truth). Use as a **trusted observation**, not something to relearn.
- **Per-person atlas (enrollment map) — exists.** Sampling the atlas at `(perp, along)` renders the line a given gaze would produce. This is the **frozen physics decoder**.
- **Perp fine channel — precise when locked.** ~6 atlas rows (sub-0.1°) during fixation, but **aliased**: many rows match → multimodal likelihood, alias spacing ≈ 126 rows ≈ 1°.
- **Perp coarse channel — localizes but blunt.** Spatial low-pass match, corr ≈ 0.81 to truth, precision ≈ 43′. Use **only** as a reacquisition anchor for far aliases; it is too crude (43′ > ~21′ real saccadic departures) to substitute for the fine channel.
- **Disambiguation physics.** Above ~820 Hz, peak-saccade motion/sample < alias spacing; separation grows to ~12× at line rate; raw single-line lock-rate holds 68–85% across the band. Split-half reliability 0.97 — the information is present.

## Architecture
**Differentiable particle filter / Monte-Carlo localization, analysis-by-synthesis:**

- **State** `x_t = (perp, along, v_perp, v_along, mode)`; `mode ∈ {pursuit, saccade}`.
- **Belief:** particle set (or mixture) over `x_t` — **multimodal, explicitly representing alias modes.** Never collapse to a unimodal Gaussian.
- **Observation model** `p(l_t | x_t)`: render the line by sampling the atlas at `(perp, along)`; score against the observed line `l_t`. Start with the physics atlas-match score; replace with a **learned, calibrated likelihood** *only if* the raw score is the bottleneck.
- **Trusted along observation:** fold the metric along-shift in directly as a low-variance measurement.
- **Dynamics — IMM, main-sequence, NOT generic smoothness.** Pursuit mode = smooth low-acceleration; saccade mode = ballistic with the peak-velocity-vs-amplitude main sequence; learned/declared mode-transition probabilities. Generic smoothness is forbidden — it rewards envelope-riding (the perp r=0.86-vs-stimulus / 0.11-vs-real-motion failure).
- **Decoder = frozen physics:** `render(x_t; atlas)` + measured per-rate noise and a saccade motion-blur model. Not learned.
- **Reacquisition (the piece that has failed 4×):** when effective sample size collapses, **reseed** from the coarse-anchor absolute position (±~43′) + along. Coarse anchor enters as an **absolute measurement**, not a veto.
- **Output a distribution per step**, not a point — preserve uncertainty for the smoother.

## Losses (self-supervised; no external trajectory needed for these)
- `L_recon` — line-scan reconstruction through the frozen atlas decoder.
- `L_dyn` — penalize off-manifold dynamics; reward main-sequence-consistent saccades; band-limit penalty (energy above a few hundred Hz).
- `L_couple` — along↔perp saccade-direction consistency (saccades ≈ straight; the trusted along component predicts the perp component, teaching the aliased axis).
- `L_anchor` *(optional, supervised)* — match the **slow component** to frame-rate full-frame-registration truth where available.

## Data
- **Train + validate the disambiguation on SYNTHETIC streams** (decisive): known trajectory from real saccade kinematics → atlas warp → inject **measured per-rate SNR + motion blur**. Gives labeled accuracy **at any rate**, including the high-rate regime no real capture can supply. Sweep rate (frame → 820 Hz → line rate).
- **Then self-supervised on REAL streams** (`L_recon + L_dyn + L_couple`).
- **Real validation:** slow component vs frame-rate truth; reproduction of oculomotor statistics (drift spectrum, microsaccade rate, main sequence); alias structure collapsing with rate as the synthetic study predicts.

## Design constraints (failure modes — do not violate)
1. **Multimodal belief is mandatory.** Unimodal collapse → gate runaway (the four worse-than-raw Kalmans).
2. **Main-sequence IMM, not smoothness.** Smoothness trains in envelope-riding.
3. **Self-consistent ≠ correct.** `L_recon + L_dyn` can converge to a smooth, plausible, *wrong* alias path. The model is only correct where the dynamics prior uniquely selects the true path — the same ≥~820 Hz condition as the velocity gate. **Prove disambiguation on synthetic (labeled); never trust self-consistency on real data as sufficiency.**
4. **Coarse anchor reseeds, never substitutes** (43′ ≫ 6′ target; > real departures).

## Implement now (ordered)
1. **Frozen physics decoder** `render(perp, along; atlas)` + measured noise/blur; verify it reconstructs held-out real lines.
2. **Synthetic generator** (known trajectory, real saccade kinematics, atlas warp, measured per-rate SNR + blur), rate-sweepable.
3. **DPF/MCL core** — multimodal particle belief, physics likelihood, trusted-along measurement, IMM main-sequence dynamics, reacquisition reseed from coarse anchor. **Delegate this; it is the repeatedly-failed component.**
4. **Train + validate on synthetic across the rate sweep** → labeled accuracy through saccades vs rate. This is the decisive disambiguation check.
5. **Add self-supervised losses** (`recon + dyn + couple`); fine-tune / deploy on real streams.
6. **Real validation:** slow vs frame-rate truth + oculomotor statistics + alias-collapse-with-rate.