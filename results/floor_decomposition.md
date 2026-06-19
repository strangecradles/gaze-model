# Horizontal floor decomposition (C1 / C2 / C3) — per subject

Phase 1 of the floor-decomposition loop. The ~4.3′ per-line horizontal `rdx_scatter`
(slow/galvo axis) is split, **per subject (never pooled)**, into:

- **C1** reference-specific error — differs across reference frames (per-frame template
  noise + each reference's own intra-frame distortion + per-reference chain error).
- **C2** reference-common systematic — argmax / sub-pixel bias, content-dependent,
  independent of which reference is used.
- **C3** real-motion **upper bound** — coherent content surviving a non-adjacent split
  (upper bound because reference-common geometric C2 still masquerades as coherent).

Method: localize each current line against chain-aligned frames N−1, N−2, N−3 **separately**
(`floor_multiref.py`, faithful replica of the committed core — j=1 reproduces committed
rdx to 0.0000 px). C1 = RMS over FOV lines of the cross-reference std. Common track =
cross-reference mean. C2 = part of the common-track residual explained by reference-
independent features (NCC peak height, peak curvature, sub-pixel phase). C3 = coherence
between same-line localizations against **different references** (`rdx1` vs `rdx2`) — the
properly non-adjacent split — plus even/odd and block splits for comparison.

## Numbers

| quantity | Igor | Ashton3 |
|---|---|---|
| baseline single-ref `rdx_scatter` | 4.33′ | 4.10′ |
| **C1** reference-specific (RMS std across N−1/2/3) | **3.91′** | **3.62′** |
| common-track residual (real + C2) | 3.46′ | 3.61′ |
| **C2** argmax/content bias (feature-explained, lower bd) | **≈0.01′** (R²=0.000) | **≈0.22′** (R²=0.004) |
| cross-reference coherence, 0–10 Hz | 0.24 | 0.30 |
| **C3** real-motion **upper bound** (√coh·common) | **≤1.70′** | **≤1.98′** |
| cross-reference coherence crossover | 21 Hz | 24 Hz |
| even/odd crossover (common track) | 6 Hz | 82 Hz |
| common-track low-freq vs machine tracker, r | −0.06 | −0.11 |
| FOV lines (all 3 refs lock) | 518 k / 1.01 M | 903 k / 1.04 M |

## Reading

**C1 dominates both subjects.** Localizations against three different reference frames
disagree by ~3.6–3.9′ RMS — comparable to the entire 4.3′ floor. The single previous
frame is a poor reference; reference-specific error is the principal term.

**C2 is negligible.** The common-track residual shows no dependence on NCC peak height,
peak curvature, or sub-pixel phase (joint R² = 0.000 Igor, 0.004 Ashton3). The "argmax /
sub-pixel bias" suspect (suspect 3 from the prior loop) is **exonerated** — it is not a
measurable component of the floor on either subject.

**Real motion is a minority and only an upper bound.** Cross-reference coherence is low
(0.24–0.30 at 0–10 Hz), so most of the floor is **not** coherent real motion. The firm
upper bound on real motion is ≤1.70′ (Igor) / ≤1.98′ (Ashton3) — and even that is inflated
by any reference-common geometric systematic. The common track barely correlates with the
machine tracker (|r| ≤ 0.11), consistent with the coherent part being small.

**The even/odd adjacency confound is confirmed.** Ashton3's even/odd crossover (82 Hz on
the common track; 118 Hz on the single-ref track in the prior loop) is **far above** the
properly non-adjacent cross-reference crossover (24 Hz). Even/odd overstates real-motion
bandwidth because adjacent lines share the frame-static atlas, so a reference systematic
appears as "coherent signal." **Consequence:** the prior loop's "refavg3 fails the signal
gate on Ashton3" verdict rested on the even/odd crossover and is therefore **not valid** —
reference averaging must be re-judged with the cross-reference / non-adjacent test.

## Classification (gates Phase 2)

| subject | dominant term | classification |
|---|---|---|
| Igor | C1 (ref-specific) ≫ real motion (≤1.70′), C2≈0 | **noise/systematic-limited — REDUCIBLE** |
| Ashton3 | C1 (ref-specific) ≫ real motion (≤1.98′), C2≈0.22′ | **noise/systematic-limited — REDUCIBLE** |

Neither subject is motion-limited; both are eligible for Phase 2.

## Caveat that reshapes Phase 2 (H2)

H2 proposes **per-line re-referencing in the particle filter** (`khz2d_methods.m4_dpf`
frame-boundary re-reference) to fix within-frame reference staleness. But this decomposition
shows the dominant, reducible term — **C1** — is **reference-specific error measured upstream
in `build_line_measurements`**, where each column is already localized in its own ±PADH
window (no per-frame staleness in the rdx metric). C1 is dominated by per-frame **template
noise** (the mechanism that made 3-frame averaging cut scatter ~12% in the prior loop), not
by within-current-frame coordinate staleness. So per-line PF re-referencing is **unlikely to
move the C1 / rdx_scatter metric**, which lives upstream of the PF.

Phase 2 will therefore (a) run the H2 power gate and A/B as specified to test it honestly,
and (b) treat the higher-value question as **re-judging multi-frame reference averaging with
the now-validated cross-reference coherence test** (since the C1 = template-noise picture and
the invalidation of the even/odd gate both point there).
