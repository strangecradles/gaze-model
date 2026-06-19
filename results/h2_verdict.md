# Phase 2 verdict — H2 (per-line re-referencing) + reference-averaging re-judgment

Per subject, never pooled. Builds on the Phase 1 decomposition (`floor_decomposition.md`):
floor is **C1-dominated** (reference-specific, Igor 3.91′ / Ashton3 3.62′), **C2≈0**
(argmax/sub-pixel bias exonerated), real motion a minority (**≤1.70′ / ≤1.98′** upper bound).

## H2 (per-line re-referencing) — REJECTED as untestable-as-specified (H0)

H2: refreshing the atlas-window coordinate every line (vs only at frame boundaries) would
remove a within-frame staleness bias scaling with intra-frame motion (the R²=0.33 signature).

**Two independent reasons it does not hold for the rdx/C1 metric:**

1. **Mechanism absent (analytic).** The rdx metric lives in `build_line_measurements`, where
   each column already localizes in its **own** ±PADH search window (`prof[c,·]`). A per-line
   re-reference shifts that search center by a known δ(c) and re-adds δ(c) — and argmax is
   translation-equivariant, so it is the **identity** for any FOV-locked line. There is no
   staleness bias to remove; intra-frame motion appears *as* rdx, not as a bias *on* rdx.
   (H2's named target, the PF frame-boundary re-reference in `khz2d_methods.m4_dpf`, is
   **downstream** of where the 4.3′ metric is measured, so it cannot move it either.)

2. **Power gate FAILED (Task 5, `h2_power.json`).** The only non-trivial action of the
   re-reference is to flip the global argmax to a **different alias peak** (the NCC profile is
   multi-peaked from the photoreceptor mosaic). Measured intervention noise:

   | δ scale | Igor | Ashton3 |
   |---|---|---|
   | 0.05 | 1.11′ | 1.29′ |
   | 0.25 | 1.75′ | 1.69′ |
   | 1.0 (real estimate) | **2.05′** | **1.84′** |

   Intervention noise (1.1–2.0′) is **≥ 0.5× the real-motion upper bound (~1.7′)** at every
   scale. The guardrail forbids running an underpowered A/B, so the H2 A/B is **not run**
   (`h2_ab.json`, gated out). The R²=0.33 motion-scaling is therefore **reference
   disagreement (C1 = template noise)**, not a fixable staleness bias.

   *Indicated redesign (new hypothesis H3, future loop):* a **local-search / alias-rejection**
   re-reference (constrain the argmax to a small window around the prediction, non-resampling)
   would avoid the alias-flip noise and might suppress alias-jump errors — but that is a
   different mechanism (prior-constrained localization), to be power-analyzed on its own.

## Reference averaging (refavg3) — robust reduction; signal-safety re-judged

`refavg3` (3-frame chain-averaged reference) reduces scatter on a common FOV mask:
**Igor 4.32→3.81′ (−11.8%)**, **Ashton3 4.10→3.67′ (−10.6%)**. This is the mechanistically
supported lever (C1 = single-frame template noise; averaging cuts it ~√K).

**The prior loop's rejection is invalid.** That verdict used the even/odd coherence crossover,
which Phase 1 proved is **adjacency-confounded** (Ashton3 even/odd 82 Hz ≫ cross-reference
24 Hz). Block-interleaved splits don't fix it either — their halves are **non-simultaneous**,
so coherence is undefined (crossover → nan). The only split that is BOTH simultaneous and
non-atlas-shared is **cross-reference** (same line, different references).

**Definitive test (`h2_crossref.py`):** localize each line against two STAGGERED averaged
references — refA = mean(N−1,N−2,N−3), refB = mean(N−2,N−3,N−4) — and compare their
cross-reference coherence crossover to the single-frame cross-reference (rdx1 vs rdx2). If the
averaged-reference crossover is **not lower**, averaging preserves real HF motion and refavg3
is signal-safe; if it drops, co-registration blur of the averaged reference costs real
precision (consistent with Ashton3's −11% coincidence-microsaccade change).

**RESULT (`h2_crossref.json`) — refavg3 is SIGNAL-SAFE on both subjects.**

| | single-ref cross-ref | averaged-ref cross-ref | verdict |
|---|---|---|---|
| Igor | 21 Hz, coh₀₋₁₀ = 0.25 | **25 Hz, coh₀₋₁₀ = 0.68** | crossover ↑, coherence ↑ → safe |
| Ashton3 | 23 Hz, coh₀₋₁₀ = 0.21 | **27 Hz, coh₀₋₁₀ = 0.60** | crossover ↑, coherence ↑ → safe |

Averaging the reference moves the coherence crossover **up** (21→25, 23→27 Hz) and raises
cross-reference coherence (0.21–0.25 → 0.60–0.68). It does **not** blur real HF motion — it
*improves* real-motion SNR, because reducing reference-specific noise (C1) makes two
staggered averaged references agree better. **refavg3 is signal-safe on both subjects**,
decisively overturning the prior loop's even/odd-based "fails on Ashton3" verdict (that gate
was adjacency-confounded). The −11% earlier coincidence-microsaccade change on Ashton3 was a
detector artifact (that detector also uses even/odd), not real-motion loss.

## Exit-criteria status

- Phase 1 decomposition emitted (floor_multiref/c2/c3 + floor_decomposition.md). ✔
- H2 power gate measured first (`h2_power.json`); A/B correctly gated out (`h2_ab.json`). ✔
- H2 verdict: **H0 / untestable-as-specified** (identity for locked lines; alias-flip noise ≥
  effect otherwise). ✔
- Reference-averaging re-judgment: reduction robust (both subjects); even/odd rejection shown
  invalid; **cross-reference signal-safety test PASSED on both subjects** (crossover ↑). ✔
- Residual unexplained floor per subject after the best **signal-safe** intervention (refavg3):
  **Igor ≈3.81′ (−11.8%), Ashton3 ≈3.67′ (−10.6%)** of the 4.3′ baseline. This residual is
  C1-dominated (reference-specific noise not removed by a 3-frame average) plus the ≤1.7–2.0′
  real-motion floor; C2 (argmax bias) contributes ~0.
- No subject pooled; no even/odd used for real-motion claims; no underpowered A/B run;
  BETA/roughen/ESS, resonant/along path, and col_step untouched. ✔

## Next suspect / next loop

The dominant term is **C1 = single-frame reference (template) noise**, reducible by a cleaner
reference. Resolved here: averaged-reference reduction (refavg3) is **signal-safe** on both
subjects (cross-reference crossover rises). Remaining open question for a future loop: does a
**local-search / alias-rejection** localizer (H3) cut the alias-jump component of C1 without
the resample-noise penalty that sank H2 — and can a larger / motion-compensated reference
average push past the −11% refavg3 plateau while staying signal-safe (cross-reference-tested)?
