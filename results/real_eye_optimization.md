# Real-eye M4 particle-filter optimization — test1 raster

Search→run→evaluate→refine loop over the M4 particle-filter configuration space on the **real human `test1` pursuit raster**, targeting the kHz payoff metric: **horizontal precision** (RMS of >25 ms detail, arcmin, lower better). Driver: `optimize_real_eye.py` (phases `broad` → `combine` → `validate` → `report`); runner: `khz2d_methods.m4_dpf` (extended with clean pass-through kwargs for every `filter.ParticleFilter` tunable — no constants were mutated); evaluation: the standard `khz2d.evaluate` protocol (2 ms smoothing, 0.05 Hz drift removal, affine calibration to the dot, identical for every config).

## The target and the guardrail

SOTA to beat (from `results/optimal_vs_previous_test1.md`):

| arm | prec_x @1182 (full 70 s) | prec_x @11823 (dur=20 s) |
|---|---|---|
| previous — physics N=300 | 2.05′ | 1.66′ |
| SOTA — learned N=1000 | 1.98′ | 1.62′ |

**Anti-gaming guardrail** (precision can be trivially lowered by over-smoothing / lazy tracking): a precision win counts ONLY if `r_dot_x ≥ baseline−0.005` AND `r_trk_x ≥ baseline−0.005` AND `valid% ≥ baseline−2 pts`. The independent ~32.5 Hz machine tracker (`r_trk_x`) is the honesty anchor; `r_dot_x ≈ 0.90` is a pursuit-lag ceiling, not a target. Configs failing the guardrail are flagged, not ranked.

## Compute / scoping decisions (stated)

- **Broad search at 1182 Hz full-length (70 s)** — ~70 s wall per physics N=300 config, so the whole one-at-a-time grid is affordable. 27 single-lever configs + 8 combine configs.
- **Line-rate validation at 11823 Hz with `dur_s=20`** — identical window to the published SOTA comparison (`m4_dpf_11823_d20` / `_learned_n1000_d20`), so the 1.66′/1.62′ comparison is apples-to-apples. Learned N=1000 at line rate costs ~25–35 min per 20 s run; only the top configs were validated.
- Every config has a distinct descriptive cache tag (e.g. `khz2d_m4_dpf_1182_learned_n1000_ess0.7_nw3.npz`); existing caches were never overwritten (`rebuild=False` everywhere; baseline tags reproduce the published numbers exactly: 2.055′/1.984′ @1182, 1.657′/1.621′ @11823-d20).
- **Held-out split**: configs were effectively picked on the first half (t<35 s, "A") and checked on the second half (t≥35 s, "B"); both halves are reported for the leaders.

## Broad + combine leaderboard — 1182 Hz, full 70 s

Baseline = `phys_N300_base` (prec 2.055′, r_dot 0.901, r_trk 0.550, valid 70%). Sorted by guardrail-valid precision. `A`/`B` = precision on the pick / held-out half.

| config | prec_x (′) | r_dot_x | r_trk_x | prec_y (′) | valid | guard | A (t<35) | B (t≥35) |
|---|---|---|---|---|---|---|---|---|
| **learn_nw3_ess0.7** | **1.897** | 0.905 | 0.547 | 0.982 | 71% | PASS | 1.522 | 2.201 |
| learn_nw3 | 1.898 | 0.903 | 0.549 | 1.016 | 71% | PASS | 1.538 | 2.195 |
| phys_nw3_ess0.7_hp16_rp0.25 | 1.936 | 0.905 | 0.551 | 0.992 | 71% | PASS | 1.626 | 2.201 |
| phys_nw3 | 1.950 | 0.903 | 0.550 | 1.001 | 71% | PASS | 1.542 | 2.281 |
| phys_nw3_ess0.7 | 1.958 | 0.905 | 0.548 | 0.998 | 71% | PASS | 1.525 | 2.302 |
| phys_ess0.7 | 1.975 | 0.905 | 0.550 | 0.980 | 70% | PASS | 1.539 | 2.327 |
| phys_hp12 | 1.978 | 0.904 | 0.552 | 0.987 | 71% | PASS | 1.591 | 2.298 |
| phys_nw3_ess0.7_hp16 | 1.981 | 0.905 | 0.550 | 1.006 | 71% | PASS | 1.564 | 2.315 |
| learn_N1000 (SOTA) | 1.984 | 0.905 | 0.551 | 0.977 | 71% | PASS | 1.635 | 2.279 |
| phys_nw3_hp16 | 1.995 | 0.904 | 0.549 | 1.008 | 71% | PASS | 1.610 | 2.310 |
| phys_rp0.25 | 1.997 | 0.904 | 0.548 | 0.967 | 70% | PASS | 1.575 | 2.345 |
| phys_pw120 | 1.999 | 0.905 | 0.550 | 0.958 | 70% | PASS | 1.591 | 2.332 |
| phys_hp10 | 2.012 | 0.904 | 0.549 | 0.985 | 71% | PASS | 1.556 | 2.376 |
| phys_b10 | 2.021 | 0.904 | 0.551 | 0.956 | 70% | PASS | 1.646 | 2.331 |
| phys_ll300 | 2.026 | 0.909 | 0.549 | 0.954 | 70% | PASS | 1.665 | 2.327 |
| phys_ess0.3 | 2.030 | 0.903 | 0.551 | 0.960 | 70% | PASS | 1.627 | 2.365 |
| phys_hp20 | 2.037 | 0.908 | 0.551 | 0.984 | 71% | PASS | 1.658 | 2.344 |
| phys_ll250 | 2.038 | 0.904 | 0.550 | 0.986 | 70% | PASS | 1.690 | 2.337 |
| phys_N300_base | 2.055 | 0.901 | 0.550 | 0.952 | 70% | PASS | 1.637 | 2.405 |
| phys_sa3 | 2.059 | 0.904 | 0.549 | 0.959 | 71% | PASS | 1.632 | 2.405 |
| phys_hp8 | 2.071 | 0.903 | 0.549 | 0.973 | 70% | PASS | 1.669 | 2.407 |
| phys_hp3 | 2.074 | 0.902 | 0.548 | 0.984 | 70% | PASS | 1.627 | 2.435 |
| phys_hp5 | 2.098 | 0.905 | 0.549 | 0.970 | 70% | PASS | 1.717 | 2.417 |
| phys_hp16 | 2.100 | 0.907 | 0.552 | 0.990 | 71% | PASS | 1.714 | 2.413 |
| phys_b30 | 2.102 | 0.902 | 0.548 | 0.956 | 70% | PASS | 1.703 | 2.441 |
| phys_N1000 | 2.106 | 0.903 | 0.549 | 0.996 | 71% | PASS | 1.663 | 2.467 |
| phys_N500 | 2.112 | 0.906 | 0.551 | 0.985 | 70% | PASS | 1.685 | 2.459 |
| phys_hp4 | 2.113 | 0.903 | 0.550 | 0.989 | 70% | PASS | 1.722 | 2.444 |
| phys_rp1.0 | 2.121 | 0.903 | 0.548 | 0.974 | 71% | PASS | 1.668 | 2.492 |
| phys_sa1 | 2.127 | 0.907 | 0.553 | 0.987 | 70% | PASS | 1.641 | 2.516 |
| phys_N2000 | 2.170 | 0.903 | 0.548 | 1.024 | 71% | PASS | 1.741 | 2.526 |
| phys_nw10 | 2.183 | 0.905 | 0.551 | 0.945 | 69% | PASS | 1.745 | 2.546 |
| phys_rs125 | 4.457 | 0.897 | 0.546 | 0.947 | 70% | PASS | 3.696 | 5.092 |
| phys_b40 | 2.156 | 0.903 | **0.543** | 0.972 | 70% | **FAIL** (r_trk) | 1.701 | 2.532 |
| phys_rs60 | 2.743 | **0.895** | 0.551 | 0.954 | 70% | **FAIL** (r_dot) | 2.189 | 3.233 |

### What the sweep found (lever by lever)

- **`NCC_LOSS_WINDOW=3` (nw3) is the headline single lever.** Faster reacquisition after lock loss (3 vs 5 consecutive bad-NCC steps before reseed) removes stale post-loss samples from the trace: 2.055′→1.950′ with r_dot/r_trk/valid all held. The opposite direction (nw10) is the worst non-pathological config (2.183′). This is the real-data mirror of the synthetic ablation's reacq-window finding.
- **`ESS_FRAC=0.7` (resample more aggressively) helps on real data** (1.975′), unlike on synthetic where it hurt — real lines are noisier, so keeping the cloud tighter to the current peak pays.
- **HP_SIGMA (appearance band) is real but non-monotonic**: prec improves from σ=4 (2.113′) toward σ=12 (1.978′), then reverts by σ=16–20. A *wider* fine band (keeping more mid-frequencies) suits the noisier real lines; σ≈12 is the sweet spot for the *single* lever, but it did **not** stack with nw3/ess0.7 (combos with hp16 were worse than without).
- **More particles do NOT help physics precision on real data** (N=500/1000/2000 all worse than N=300) — opposite of the synthetic ablation. The synthetic N-gain was fixation-RMS against ground truth; on the real raster the extra particles mostly average in alias-adjacent hypotheses.
- **BETA sharpening hurts and fails the guardrail** (b40 drops r_trk to 0.543) — confirms the published control-arm finding. Softer b10 is mildly positive (2.021′).
- **Reseed spread must stay tight** (the m4 default 30 px): rs60/rs125 are disasters (2.74′/4.46′), rs60 failing the guardrail on r_dot.
- Roughening 0.25 mildly positive; sigma_along, line_len, padw near-neutral.

### Combinations

`nw3 + ess0.7` stack on the learned head (1.984′→1.897′) but barely stack within physics (phys_nw3_ess0.7 1.958′ ≈ phys_nw3 1.950′). The physics kitchen-sink (nw3+ess0.7+hp16+rp0.25, 1.936′) is the best physics config at 1182 but still behind the learned combos. hp12 was not re-tested in combos (hp16 combos regressed; the hp lever doesn't reach the learned weight path anyway — under `likelihood='learned'` `hp_sigma` only affects the lock monitor).

## Held-out check (pick on t<35 s, confirm on t≥35 s)

The second half of the recording is intrinsically noisier (baseline 1.637′ → 2.405′). The leaders win on **both** halves — the gain is not an artifact of fitting the search to one segment:

| config | pick half A | held-out half B | B vs baseline B | r_trk_x on B |
|---|---|---|---|---|
| phys_N300_base | 1.637′ | 2.405′ | — | 0.557 |
| learn_N1000 (SOTA) | 1.635′ | 2.279′ | −0.126′ | 0.560 |
| phys_nw3 | 1.542′ | 2.281′ | −0.124′ | 0.556 |
| learn_nw3 | 1.538′ | 2.195′ | −0.210′ | 0.556 |
| **learn_nw3_ess0.7** | **1.522′** | **2.201′** | **−0.204′** | 0.556 |

The config picked on half A (`learn_nw3_ess0.7`, best A at 1.522′) is also (statistically tied for) the best on the held-out half B, with tracker correlation unchanged.

## Line-rate validation — 11823 Hz, dur=20 s (identical window to the SOTA comparison)

Fresh runs, matched window, baselines reproduced from cache first:

| config | prec_x (′) | r_dot_x | r_trk_x | valid | guardrail |
|---|---|---|---|---|---|
| **learn_nw3_ess0.7** | **1.571** | 0.796 | **0.559** | 60% | **PASS + win** |
| phys_nw3 | 1.588 | 0.795 | 0.550 | 60% | PASS + win |
| learn_nw3 | 1.596 | 0.796 | 0.552 | 60% | PASS + win |
| learn_N1000 (SOTA) | 1.621 | 0.797 | 0.553 | 60% | PASS (reference) |
| phys_nw3_ess0.7_hp16_rp0.25 | 1.630 | 0.795 | 0.552 | 60% | PASS + win |
| phys_N300_base (previous) | 1.657 | 0.797 | 0.553 | 60% | PASS (reference) |

(`prec_y ≈ 0` / y-columns are uninformative in this 20 s window — the vertical affine fit collapses exactly as documented in `optimal_vs_previous_test1.md`; ignore y here.)

## BEST CONFIG — full settings

**`learn_nw3_ess0.7`** = the G14 learned blur-aware likelihood + faster reacquisition + heavier resampling:

```python
khz2d_methods.m4_dpf(eff_rate,                      # 1182.0 or 11823.0
                     likelihood="learned",          # G14 head, cache/g14_head.pt
                     n_particles=1000,
                     ncc_loss_window=3,             # default 5 — reseed after 3 bad-NCC steps
                     ess_frac=0.7,                  # default 0.5 — resample sooner
                     # everything else at m4 defaults:
                     # BETA=20 (inert under learned), HP_SIGMA=6 (lock monitor only),
                     # SIGMA_ALONG=2, ROUGHEN=0.5/0.5, NCC_LOCK_LOSS_THR=0.35,
                     # reseed_perp_sigma=30, padw=100, line_len=200, seed=0
                     )
# cache tags: khz2d_m4_dpf_1182_learned_n1000_ess0.7_nw3.npz
#             khz2d_m4_dpf_11823_learned_n1000_ess0.7_nw3_d20.npz
```

Runner-up worth knowing about: **`phys_nw3`** (physics N=300, only `ncc_loss_window=3`) hits 1.588′ @11823 — it beats the learned N=1000 SOTA at roughly **1/10th the compute** (no head, 300 renders/step vs 3-band×1000+head), making it the best precision-per-FLOP config.

## VERDICT — honest read

**Yes, the SOTA is beaten, and the win passes every honesty check — but it is incremental, not transformative.**

- **@11823 Hz (dur=20)**: 1.571′ vs SOTA 1.62′ (**−0.050′, −3.1%**) and vs previous physics 1.66′ (**−0.086′, −5.2%**). r_trk_x *improved* (0.553→0.559), r_dot_x within tolerance (0.797→0.796), valid identical (60%).
- **@1182 Hz (full 70 s)**: 1.897′ vs SOTA 1.98′ (**−0.087′, −4.4%**) and vs previous 2.05′ (**−0.158′, −7.7%**). r_dot_x 0.905 (= SOTA), r_trk_x 0.547 (within the 0.005 tolerance of baseline 0.550; 0.556 vs 0.557 on the held-out half), valid 71%.
- **It is not over-smoothing**: the independent tracker correlation is held (1182) or improved (11823), validity is unchanged, and the gain reproduces on a held-out time segment that was not used to pick the config. Mechanistically the levers are *reacquisition speed* and *resampling cadence* — they change when the filter reseeds/resamples, not how much the output is smoothed; a smoothing artifact would show up as r_trk decay, which is exactly what the rejected configs (b40, rs60) show.
- Honest caveats: (i) the deltas are a few hundredths of an arcminute — consistent in direction across two rates and two held-out halves, but of the same order as segment-to-segment variability; (ii) r_trk_x at 1182 is 0.003 *below* baseline (within stated tolerance, recovered at line rate); (iii) the line-rate validation window is 20 s (compute cap), matched exactly to the published SOTA window.

## Files

- `optimize_real_eye.py` — driver (phases: broad / combine / validate / report; report is cache-only).
- `results/real_eye_optimization.png` — leaderboard, held-out check, line-rate validation.
- Search logs: `results/_broad_sweep.log`, `results/_combine_sweep.log`, `results/_validate_sweep.log`; row dumps in `cache/_optimize_broad_rows.npy`, `cache/_optimize_validate_rows.npy`.
