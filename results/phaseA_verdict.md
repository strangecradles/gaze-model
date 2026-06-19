# Phase A — re-metric + locate the flips (per subject, never pooled)

Metric of record: **flip-rate + core-RMS on the PF OUTPUT** (never std). Labels are
cross-reference (lines_multiref3), never even/odd.

## Population split (FOV lines)

| label | Igor | Ashton3 | definition |
|---|---|---|---|
| alias-flip-prone | 46.1% | 33.3% | refs N−1/2/3 disagree ≥3px (mosaic inconsistent) |
| real-microsaccade | 3.0% | 3.4% | refs AGREE on a ≥3px jump (eye moved); main-seq r=0.30/0.35 |
| clean | 30.2% | 42.0% | no disagreement, no jump (sub-px core) |

## Metric: harness rdx vs PF output

| | flip-rate | core-RMS | microsacc-preserved |
|---|---|---|---|
| Igor harness rdx | 0.522 | 0.515′ | 0.91 |
| **Igor PF output** | **0.393** | 1.159′* | 0.77 |
| Ashton3 harness rdx | 0.518 | 0.483′ | 0.92 |
| **Ashton3 PF output** | **0.401** | 0.856′* | 0.79 |

\* PF core-RMS is partly inflated by **track-source bias** (the flip-resistant track is
multiref-derived, so the harness family sits closer to it by construction). This bias is
common to ON/OFF, so it cancels in the Phase B A/B; do not read the absolute harness↔PF
core-RMS gap as a real PF penalty.

## CRUX — do the flips survive the PF? **YES.**

PF flip-rate (0.39–0.40) is only ~25% below the harness (0.52) — **the PF does not suppress
the mosaic flips; ~40% of flip-prone lines still depart ≥3px in the deliverable.** The
autopsy (`pf_flip_autopsy.json`) shows why: at flip-prone lines the PF output is

| handling | Igor | Ashton3 |
|---|---|---|
| reject / clean (<1.5px) | 0.40 | 0.39 |
| **bimodal-middle (1.5–4.5px, averaged-in)** | 0.32 | 0.32 |
| followed the flip (>4.5px) | 0.28 | 0.28 |

median departure ~2.1px. The PF's alias-robust weighted mean uses a **30′** window
(0.5×ALIAS_SPACING_ROWS), tuned to the coarse ~1° alias; the mosaic flip is **2.86′**, far
inside the window, so the robust mean **averages the flipped particles in** (32%) or follows
the flipped peak (28%) instead of rejecting it. The PF is structurally blind to the mosaic
alias — confirmed analytically and empirically.

## Reducible vs irreducible

- **Reducible:** the alias-flip population reaching the PF output (flip-rate ≈0.39–0.40 at
  the 46%/33% flip-prone lines). These are cross-reference-INCONSISTENT — provably not real
  eye motion.
- **Irreducible:** real microsaccades (~3% of FOV lines, cross-reference-CONSISTENT,
  main-sequence-following) + the true clean core-RMS (~0.5′). These must be preserved.

## Verdict → proceed to Phase B

The flips survive into the deliverable and the PF's existing alias machinery cannot reach the
mosaic scale. Phase B (mode-gated mosaic-scale rejection INSIDE the PF) is warranted. The
signal-safety constraint is sharp: real microsaccades (3%) are legitimate ~mosaic-spacing
jumps, so rejection MUST be mode-gated (relaxed in saccade mode) and validated by
cross-reference + main-sequence preservation, never an even/odd detector.

Artifacts: `flip_labels.json`, `metric_rdx_vs_pf.json`, `pf_flip_autopsy.json`.
