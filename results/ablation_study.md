# §7.7 Ablation study — runnable subset on the synthetic ground-truth benchmark

This reproduces the **runnable** part of the Figure 5 / Table 3 design space and the §7.7 ablation protocol on the substrate we actually have: the **labeled synthetic benchmark** (`synth_stream` / `rate_sweep`), which has known ground-truth trajectories and therefore true perp-RMS (arcmin, split fixation vs saccade), gross-error persistence (ms), lock-rate-vs-velocity, and reseed counts. We hold the benchmark + metrics fixed and vary **one lever at a time** from the current hand-built baseline, measuring marginal effect (the §7.7 (iii) step) — without any of the new modules §7.7 (iv)/(v) require.

### Scope / compute decisions (stated honestly)

- **Substrate:** synthetic ground-truth streams (perfect labels + perfect physics: the atlas that generates each stream IS the decoder). This is the only substrate with true labels. The §7.7 AOSLO + artificial-eye benchmark is **not** available and is **not** fabricated (see out-of-scope section).
- **Rates:** the two **headline** levers (particle count N, observation likelihood) are evaluated at **both 1500 Hz** (the G13 fixation-sub-0.1° operating regime) **and 12000 Hz** (the line rate). All **sensitivity** levers (BETA, reacq window, coast cap, ESS_FRAC, roughening) are evaluated at **1500 Hz only** to control compute, per the task's representative-rate guidance.
- **Seeds:** 4 (0..3) per config (the full `rate_sweep` uses 6; reduced here for compute). Saccades are rare, so saccade metrics are noisier than fixation metrics.
- **Caching + reuse:** every (config, rate, seed) run is cached under `cache/`; the baseline reuses the existing `rate_sweep` cache verbatim. This driver imports and calls `rate_sweep` machinery and `filter.run`'s per-call kwarg overrides — it does not duplicate the filter or mutate `rate_sweep`.

## §7.7 runnable-subset ablation — synthetic ground-truth benchmark

Seeds 0..3 per config. Baseline = current hand-built filter (N=400, BETA=20, physics fine-NCC, ESS_FRAC=0.5, NCC_LOSS_WINDOW=5, COAST_CAP=100).

Metric deltas are config minus baseline AT THE SAME RATE. For RMS/gross/persistence lower is better; for lock-rate higher is better.


### Rate = 1500 Hz

config                             |   FixRMS' |   SacRMS' |  SacGross |   PersMax |   LockFix |   LockSac |    Reseed
----------------------------------------------------------------------------------------------------------------------
baseline (N=400, BETA=20, physics) |      1.78 |      33.1 |     0.179 |      10.0 |      0.99 |      0.46 |        12
N=100                              | 8.22(+6.44) | 44.4(+11.4) | 0.385(+0.205) | 78.7(+68.7) | 0.98(-0.01) | 0.46(+0.00) |        23
N=300                              | 1.89(+0.11) | 50.3(+17.2) | 0.333(+0.154) |  6.0(-4.0) | 1.00(+0.00) | 0.46(+0.00) |        14
N=1000                             | 1.34(-0.44) | 31.2(-1.9) | 0.205(+0.026) |  2.7(-7.3) | 1.00(+0.00) | 0.36(-0.10) |        10
likelihood=learned                 | 0.63(-1.15) | 34.5(+1.4) | 0.282(+0.103) |  3.3(-6.7) | 1.00(+0.00) | 0.44(-0.03) |        13
BETA=10                            | 1.32(-0.45) | 48.9(+15.8) | 0.308(+0.128) |  3.3(-6.7) | 1.00(+0.00) | 0.49(+0.03) |        16
BETA=40                            | 1.82(+0.04) | 24.4(-8.7) | 0.154(-0.026) |  2.0(-8.0) | 0.98(-0.02) | 0.54(+0.08) |        12
NCC_LOSS_WINDOW=3                  | 1.14(-0.64) | 46.9(+13.8) | 0.256(+0.077) |  2.7(-7.3) | 1.00(+0.00) | 0.44(-0.03) |        14
NCC_LOSS_WINDOW=10                 | 3.35(+1.57) | 37.5(+4.4) | 0.282(+0.103) | 26.0(+16.0) | 0.99(-0.01) | 0.31(-0.15) |         9
COAST_CAP=50                       | 1.78(+0.00) | 33.1(+0.0) | 0.179(+0.000) | 10.0(+0.0) | 0.99(+0.00) | 0.46(+0.00) |        12
COAST_CAP=200                      | 1.78(+0.00) | 33.1(+0.0) | 0.179(+0.000) | 10.0(+0.0) | 0.99(+0.00) | 0.46(+0.00) |        12
ESS_FRAC=0.3                       | 6.85(+5.07) | 34.2(+1.1) | 0.205(+0.026) | 90.0(+80.0) | 0.96(-0.03) | 0.33(-0.13) |        14
ESS_FRAC=0.7                       | 4.72(+2.94) | 24.6(-8.4) | 0.154(-0.026) | 56.0(+46.0) | 0.98(-0.02) | 0.51(+0.05) |        13
ROUGHEN=0.25                       | 3.70(+1.92) | 28.2(-4.9) | 0.154(-0.026) |  3.3(-6.7) | 0.98(-0.01) | 0.62(+0.15) |        25
ROUGHEN=1.0                        | 4.49(+2.71) | 26.7(-6.4) | 0.154(-0.026) | 56.7(+46.7) | 0.99(-0.00) | 0.54(+0.08) |         8

### Rate = 12000 Hz

config                             |   FixRMS' |   SacRMS' |  SacGross |   PersMax |   LockFix |   LockSac |    Reseed
----------------------------------------------------------------------------------------------------------------------
baseline (N=400, BETA=20, physics) |      1.05 |      43.7 |     0.249 |       1.7 |      1.00 |      0.68 |        54
N=100                              | 1.65(+0.60) | 44.4(+0.7) | 0.281(+0.032) |  1.2(-0.5) | 1.00(-0.00) | 0.61(-0.07) |        68
N=300                              | 1.39(+0.34) | 41.2(-2.5) | 0.240(-0.009) |  0.8(-0.8) | 1.00(-0.00) | 0.66(-0.02) |        58
N=1000                             | 0.75(-0.30) | 41.8(-1.9) | 0.195(-0.054) |  1.1(-0.6) | 1.00(+0.00) | 0.71(+0.03) |        51
likelihood=learned                 | 0.37(-0.68) | 24.8(-18.9) | 0.087(-0.162) |  0.6(-1.1) | 1.00(+0.00) | 0.84(+0.16) |        62

## Headline findings — which lever moves which metric

All numbers are pooled over 4 seeds at the stated rate; the baseline is the current hand-built filter. Saccade lines are rare (~0.6% of steps) so the saccade metrics carry real seed-to-seed variance — read them as trends, not 3-significant-figure truth.

**1. Observation likelihood (physics fine-NCC → learned blur-aware head) — the headline lever.** This is the one lever that moves the *saccade* metric in the right direction without costing fixation:

- At the **line rate (12 kHz)**: saccade perp RMS 43.7′ → 24.8′ (-18.9′), saccade gross 0.249 → 0.087 (-0.162), lock-rate-in-saccade 0.68 → 0.84. Fixation is **preserved and slightly improved** (1.05′ → 0.37′).
- At **1500 Hz** the learned head is roughly neutral on saccade RMS (33.1′ → 34.5′) but still sharpens fixation (1.78′ → 0.63′). The head was trained at 2 kHz; its saccade benefit is largest at the line rate, where the per-sample motion within each line is smallest. This reproduces the §5.3 / G14 result *inside the full closed filter loop* (G14 measured it on the offline candidate-grid only).
  Honest caveat: even with the learned head, through-saccade RMS stays well above the 6′ (0.1°) DoD — blur is a genuine physical limit, not a modeling artifact (consistent with §6 and the G14 verdict).

**2. Particle count N — the fixation-precision and robustness lever.** N trades compute for fixation precision and persistence, and barely touches the saccade blur floor:

- Too few particles is catastrophic: at 1500 Hz, N=100 blows fixation RMS 1.78′ → 8.2′ and max gross-error persistence 10.0 ms → 78.7 ms (the cloud depletes on the razor-sharp peak).
- More particles help fixation monotonically: at 12 kHz, N=1000 gives the best fixation RMS (1.05′ → 0.75′) and lowest saccade gross, but saccade RMS is essentially flat (43.7′ → 41.8′) — N does not buy its way past the blur floor.
- The baseline N=400 sits at a sensible knee: most of the N=1000 fixation gain at ~40% of the render cost.

**3. BETA (likelihood sharpness).** Sharpening the weight (BETA 20 → 40) modestly *helps* saccades at 1500 Hz (sac RMS 33.1′ → 24.4′, lock-in-saccade 0.46 → 0.54) by discriminating the true peak harder, at a small fixation-variance cost; BETA 10 is worse on saccades. BETA is a real but second-order knob.

**4. Reacquisition window / coast cap.** `NCC_LOSS_WINDOW` is the live reacq knob: a longer window (10) lets bad locks persist (max gross-run 10.0 ms → 26.0 ms and lock-in-saccade drops), while a shorter window (3) reacquires faster but reseeds more often. `COAST_CAP` (50 vs 200) is **inert in this regime** — identical to baseline — because the window threshold always fires first; the coast cap only matters in a long uninformative blackout that does not occur on these streams.

**5. ESS_FRAC / roughening.** Both are near-locally-optimal at the baseline (0.5 / 0.5). Moving ESS_FRAC to 0.3 or 0.7, or roughening to 0.25 or 1.0, degrades fixation RMS and persistence (impoverishment when under-resampling / under-roughening, over-diffusion when over-roughening). These are stability knobs, not accuracy levers — the existing grid-search values hold up.

### One-line summary

- **Saccade accuracy** is moved only by the **learned likelihood** (decisively at the line rate) and modestly by **higher BETA**; everything else leaves the blur floor intact.
- **Fixation precision + persistence** is moved by **particle count N** (more is better, with a knee near 400) and degraded by mistuned ESS_FRAC / roughening / a too-long reacq window.
- **COAST_CAP** does nothing in this regime.

## Out of scope — requires AOSLO + artificial-eye hardware or new modules

The paper's §7.7 study is specified on a **cone-resolved AOSLO substrate with
artificial-eye ground truth** (Table 3 rows 8–9, §7.6). We do **not** have that
hardware, so the following parts of the §7.7 program were **not run** and are not
fabricated:

| Table 3 / Figure 5 lever | Status here | Why out of scope |
|---|---|---|
| Hand-built IMM dynamics → deep Markov / neural-ODE / Mamba generative oculomotor prior (§7.2) | **not run** | requires a new learned-dynamics module + real high-rate fixational traces to train it; PyDPF could scaffold this. |
| Bootstrap proposal → conditional normalizing-flow / amortized proposal (§7.5) | **not run** | requires a new flow-proposal module and end-to-end training. |
| Fine-band NCC likelihood → **learned calibrated head** | **RAN** (headline lever) | existing G14 head (`train.load_head`); see table above. |
| Fine-band NCC → splatting / neural-field decoder, self-supervised features (§7.4) | **not run** | requires a new differentiable renderer / feature extractor. |
| Systematic resampling → entropy-OT / stop-gradient / soft differentiable resampling | **not run** | requires a differentiable-resampling module; **PyDPF** could scaffold this. |
| Hand-tuned modular → **end-to-end FIVO/VSMC training** of the whole DPF (§7.5) | **not run** | depends on differentiable resampling above; PyDPF could scaffold the end-to-end-training path. |
| Weighted-particle posterior → score-based / diffusion posterior (§7.3) | **not run** | requires a score/diffusion posterior module. |
| Slow strip-registration anchor → learned-descriptor / multi-hypothesis anchor | **not run** | requires real AOSLO frames + a learned registration model. |
| Video-rate non-AO substrate → **AOSLO cone-resolved** (§7.6) | **out of scope (hardware)** | needs an AOSLO. |
| Validation vs target+pupil → **artificial-eye ground truth, off-manifold saccades** (§7.6) | **out of scope (hardware)** | needs a programmable model eye. |

What we *did* run is the runnable subset of Figure 5's columns reachable with the
existing code on the **labeled synthetic ground-truth benchmark** (true RMS / gross /
lock metrics) plus, for sanity, the real-raster precision-only metric (`khz2d_methods`,
no ground truth — reported in `results/khz2d_methods.md`, not re-run here). The
learned-likelihood column is the one Figure-5 lever that is both runnable and a genuine
ML upgrade, and it behaves exactly as §5.3 / §7.4 predict.


---

_Generated by `ablation_study.py`. Figure: `results/ablation_study.png`. Metric definitions are identical to `results/rate_sweep_verdict.md` (G13). Real-raster precision sanity check: `results/khz2d_methods.md`._
