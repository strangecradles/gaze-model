# C1 composition — measured (Phase A), and why Phase B is gated out

The reference-specific floor **C1** (dominant term of the 4.3′ horizontal floor; Igor 3.91′,
Ashton3 3.62′) is decomposed into measured sub-components, **per subject, never pooled**. This
**replaces the previously assumed "template noise" label** — which is now refuted.

Method (`c1_composition.py`, on the `lines_multiref3` cache): per line, `dev(L,k) = rdx(L,k) −
mean_k rdx(L,·)` across references N−1/N−2/N−3; `V` = pooled variance of `dev` over FOV (L,k).

## Measured composition (variance shares of C1; NON-orthogonal)

| sub-component | Igor | Ashton3 | how measured |
|---|---|---|---|
| **ALIAS jumps** (mosaic-peak flips, \|dev\|>3px) | **0.98** | **0.98** | heavy-tail variance share |
| CHAIN error (age-gap) | 0.25 | 0.33 | var(rdx_i−rdx_j) slope vs hop count |
| DISTORTION (ref intra-frame motion) | **0.00** | **0.00** | core \|dev\| vs ref \|inc\|, R²=0.02/0.03 |
| TEMPLATE (low-SNR, motion-indep.) | **0.00** | **0.00** | core dev² vs 1/NCC-peak-height |

(Fractions are attributions, not a partition — alias and chain overlap because large-age-gap
reference pairs flip more often. Alias is the unambiguous dominant term.)

## What this means

1. **C1 is ~98% ALIAS-JUMPS** — the NCC localizer selecting a *different photoreceptor-mosaic
   peak* when the current line is matched against different reference frames. The dev
   distribution is heavy-tailed (|dev|>3px = 24%/17% of (L,k) but ~98% of the variance), with a
   ~6px discrete jump scale = the mosaic spacing.
2. **"Template noise" is REFUTED.** The motion-independent, SNR-correlated core (the actual
   template-noise signature) carries ≈0 of C1.
3. **Smooth intra-frame DISTORTION of references is ≈0** (core dev² vs reference motion
   R²=0.02–0.03). The floor's R²=0.33 motion-scaling is **not** smooth distortion — it is the
   **alias-jump RATE being motion-driven** (per-frame alias rate vs |inc|: R²=0.13 Igor / 0.17
   Ashton3, positive slope). More eye motion → references show more different retinal content →
   more mosaic ambiguity → more flips.
4. **CHAIN error contributes ~25–33%** (2-hop reference pairs disagree more than 1-hop), but is
   entangled with alias (older references flip more).

## Phase B (per-reference dewarp) — GATED OUT (not run)

The objective runs Phase B "only if the distortion fraction is non-trivial." **It is ≈0.**
Per-reference dewarp corrects a *smooth* within-reference warp; that component carries
essentially none of C1, so dewarping references cannot reduce the alias-dominated floor. The
guardrail (don't run an intervention against a ≈0 effect) applies — **Phase B and Phase C are
not run.** This is consistent with H1 (current-frame atlas dewarp) and H2 (re-reference) both
being dead: none of the three dewarp/re-reference interventions can touch an alias-jump floor.

## Redirected next lever (H3)

The productive, measured lever is **alias-jump rejection in the localizer**, not reference
geometry: constrain the per-line NCC argmax to a small window around a causal motion prediction
(local search) so it cannot flip to a far mosaic peak — a *non-resampling* operation, avoiding
H2's resample/alias-flip penalty. The decomposition predicts the recoverable headroom is the
alias share (~0.98 of C1 ≈ most of the 3.6–3.9′), making this the highest-value next experiment.
It also explains refavg3's −11% (averaged reference has a cleaner single peak → fewer flips) and
the refavg5<refavg3 plateau (stale frames add content mismatch → more flips).

## Exit status

- Phase A emitted `c1_composition.json` (per subject) + this verdict naming the dominant
  sub-component (**alias**), replacing the assumed "template noise." ✔
- Phase B: distortion fraction ≈0 → correctly **not run** (gated). ✔
- No subjects pooled; no even/odd used; no underpowered/zero-effect A/B run; BETA/roughen/ESS,
  resonant/along path, col_step untouched. ✔
