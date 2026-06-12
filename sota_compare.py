"""sota_compare.py — apples-to-apples head-to-head: the faithful Stevenson &
Roorda composite-reference strip SOTA vs OUR particle filter (M4), physics-M4,
and incremental strips (M1), on the SAME test1 raster and the SAME khz2d
evaluation protocol. Writes results/sota_comparison.{md,png}.

Run AFTER sota_strip.py has cached sota_s8 and sota_s1_d20.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import khz2d

RESULTS = khz2d.RESULTS


def _ev(tag, label, rate=None, tmax=None, ncc_thr=None):
    z = khz2d.load_method(tag)
    if z is None:
        return None
    t = z["t"]; x = z["x_px"]; y = z["y_px"]; v = z["valid"].astype(bool)
    if ncc_thr is not None and "max_ncc" in z:        # re-threshold a SOTA run
        v = z["max_ncc"] > ncc_thr
    if tmax is not None:                              # restrict to a common window
        m = t <= tmax
        t, x, y, v = t[m], x[m], y[m], v[m]
    r = float(rate if rate is not None else z["rate"])
    ev = khz2d.evaluate(t, x, y, v, r, label, smooth_ms=2)
    return ev


def _thr_for_valid(tag, target, tmax=None):
    """ncc_thr that yields ~target valid fraction for a cached SOTA run."""
    z = khz2d.load_method(tag)
    ncc = z["max_ncc"]
    if tmax is not None:
        ncc = ncc[z["t"] <= tmax]
    return float(np.quantile(ncc, 1.0 - target))


def main():
    rows = []   # (group, ev, note)

    # ---- ~1478 Hz tier (SOTA S=8 <-> our 1182 Hz M4, full 70 s) ----
    # SOTA at the same numeric strip-NCC acceptance as incremental M1 (0.35),
    # and at a validity-matched threshold (so precision is apples-to-apples).
    thr_match8 = _thr_for_valid("sota_s8", 0.71)
    rows.append(("~1.5 kHz", _ev("sota_s8", "SOTA composite S=8 (thr0.35)",
                                 ncc_thr=0.35), "faithful thr"))
    rows.append(("~1.5 kHz", _ev("sota_s8", "SOTA composite S=8 (valid-matched)",
                                 ncc_thr=thr_match8), f"thr={thr_match8:.2f}"))
    rows.append(("~1.5 kHz", _ev("m1_s8", "Incremental strips M1 S=8", rate=1478.0),
                 "prev-frame ref"))
    rows.append(("~1.5 kHz", _ev("m4_dpf_1182_learned_n1000_ess0.7_nw3",
                                 "OUR best PF (learned) @1182"), "M4 best"))
    rows.append(("~1.5 kHz", _ev("m4_dpf_1182", "OUR physics M4 @1182"),
                 "M4 baseline"))

    # ---- 11823 Hz tier (line rate, 20 s window for apples-to-apples) ----
    thr_match1 = _thr_for_valid("sota_s1_d20", 0.60, tmax=20.0)
    rows.append(("11.8 kHz", _ev("sota_s1_d20", "SOTA composite S=1 (thr0.35)",
                                 tmax=20.0, ncc_thr=0.35), "faithful thr"))
    rows.append(("11.8 kHz", _ev("sota_s1_d20", "SOTA composite S=1 (valid-matched)",
                                 tmax=20.0, ncc_thr=thr_match1), f"thr={thr_match1:.2f}"))
    rows.append(("11.8 kHz", _ev("m1_s1", "Incremental strips M1 S=1", rate=11823.0,
                                 tmax=20.0), "prev-frame ref"))
    rows.append(("11.8 kHz", _ev("m4_dpf_11823_learned_n1000_ess0.7_nw3_d20",
                                 "OUR best PF (learned) @11823"), "M4 best"))
    rows.append(("11.8 kHz", _ev("m4_dpf_11823_d20", "OUR physics M4 @11823"),
                 "M4 baseline"))

    rows = [(g, e, n) for (g, e, n) in rows if e is not None]

    # ---- console + markdown leaderboard ----
    lines = []
    for g, e, n in rows:
        lines.append(khz2d.summarize(e) + f"   [{g}; {n}]")
    print("\n".join(lines))

    _write_md(rows, thr_match8, thr_match1)
    _write_png(rows)


def _row_md(e, n):
    return (f"| {e['label']} | {e['rate']:.0f} | {e['r_dot_x']:.3f} | "
            f"{e['r_trk_x']:.3f} | {e['rms_x']:.1f} | {e['prec_x']:.2f} | "
            f"{e['prec_y']:.2f} | {e['valid_frac']*100:.0f}% | {n} |")


def _write_md(rows, thr8, thr1):
    P = os.path.join(RESULTS, "sota_comparison.md")
    g1 = [(e, n) for g, e, n in rows if g == "~1.5 kHz"]
    g2 = [(e, n) for g, e, n in rows if g == "11.8 kHz"]
    hdr = ("| method | rate (Hz) | r_dot_x | r_trk_x | RMS_x (′) | "
           "prec_x (′) | prec_y (′) | valid | note |\n"
           "|---|---|---|---|---|---|---|---|---|")

    def by(es, key):
        d = {e["label"]: e for e, _ in es}
        return d

    with open(P, "w") as fh:
        fh.write(_DOC.format(thr8=thr8, thr1=thr1))
        fh.write("\n\n## Head-to-head — ~1.5 kHz tier (full 70 s)\n\n")
        fh.write(hdr + "\n")
        for e, n in g1:
            fh.write(_row_md(e, n) + "\n")
        fh.write("\n## Head-to-head — 11.8 kHz line-rate tier (20 s window)\n\n")
        fh.write(hdr + "\n")
        for e, n in g2:
            fh.write(_row_md(e, n) + "\n")
        # verdict deltas
        d1 = by(g1, 0); d2 = by(g2, 0)
        sota1 = d1.get("SOTA composite S=8 (valid-matched)")
        m4b1 = d1.get("OUR best PF (learned) @1182")
        sota2 = d2.get("SOTA composite S=1 (valid-matched)")
        m4b2 = d2.get("OUR best PF (learned) @11823")
        m1a = d1.get("Incremental strips M1 S=8")
        fh.write("\n## Verdict — what our method buys over the SOTA, on our data\n\n")
        if sota1 and m4b1:
            fh.write(_delta("~1.5 kHz / 1182 Hz", sota1, m4b1, m1a))
        if sota2 and m4b2:
            fh.write(_delta("11.8 kHz line rate (20 s)", sota2, m4b2, None))
    print("wrote", P)


def _delta(rate_label, sota, m4, m1):
    dp = sota["prec_x"] - m4["prec_x"]
    dt = m4["r_trk_x"] - sota["r_trk_x"]
    s = (f"- **{rate_label}:** OUR PF precision_x **{m4['prec_x']:.2f}′** vs SOTA "
         f"**{sota['prec_x']:.2f}′** at matched validity "
         f"(**Δ = {dp:+.2f}′**, {100*dp/sota['prec_x']:+.0f}% — "
         f"{'PF better' if dp>0 else 'SOTA better'}). "
         f"r_trk_x PF {m4['r_trk_x']:.3f} vs SOTA {sota['r_trk_x']:.3f} "
         f"(Δ {dt:+.3f}). r_dot_x PF {m4['r_dot_x']:.3f} vs SOTA "
         f"{sota['r_dot_x']:.3f}. valid PF {m4['valid_frac']*100:.0f}% vs SOTA "
         f"{sota['valid_frac']*100:.0f}%.\n")
    if m1 is not None:
        dpi = m1["prec_x"] - sota["prec_x"]
        s += (f"  - Composite reference vs incremental strips: SOTA "
              f"{sota['prec_x']:.2f}′ vs M1 {m1['prec_x']:.2f}′ "
              f"(**{dpi:+.2f}′** — composite "
              f"{'better' if dpi>0 else 'worse'}; validates the SOTA's key idea).\n")
    return s


def _write_png(rows):
    P = os.path.join(RESULTS, "sota_comparison.png")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use("default")
    colors = {"SOTA": "#d1495b", "Incremental": "#edae49",
              "OUR best": "#2e86ab", "OUR physics": "#5fa8d3"}

    def cof(lbl):
        if lbl.startswith("SOTA"):
            return "#d1495b" if "matched" in lbl else "#f1a7b4"
        if lbl.startswith("Incremental"):
            return "#edae49"
        if "best" in lbl:
            return "#1b4965"
        return "#5fa8d3"

    for k, (g, title) in enumerate([("~1.5 kHz", "~1.5 kHz tier (full 70 s)"),
                                    ("11.8 kHz", "11.8 kHz line rate (20 s)")]):
        es = [(e, n) for gg, e, n in rows if gg == g]
        labels = [e["label"].replace(" composite", "").replace(" strips", "")
                  for e, _ in es]
        prec = [e["prec_x"] for e, _ in es]
        rtrk = [e["r_trk_x"] for e, _ in es]
        valid = [e["valid_frac"] * 100 for e, _ in es]
        cols = [cof(e["label"]) for e, _ in es]
        xpos = np.arange(len(es))
        bars = ax[k].bar(xpos, prec, color=cols, edgecolor="black", lw=0.7)
        for j, (p, rt, vv) in enumerate(zip(prec, rtrk, valid)):
            ax[k].text(j, p + 0.05, f"{p:.2f}′\nr_trk={rt:.2f}\nv={vv:.0f}%",
                       ha="center", va="bottom", fontsize=8)
        ax[k].set_xticks(xpos)
        ax[k].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax[k].set_ylabel("horizontal precision (arcmin, lower=better)")
        ax[k].set_title(title)
        ax[k].grid(axis="y", alpha=0.3)
        ax[k].set_ylim(0, max(prec) * 1.35)
    fig.suptitle("Composite-reference SOTA (Stevenson–Roorda) vs our particle "
                 "filter — same test1 raster, same protocol", fontsize=13)
    fig.tight_layout()
    fig.savefig(P, dpi=130)
    print("wrote", P)


_DOC = """# SOTA head-to-head — Stevenson & Roorda composite-reference strip registration vs our particle filter (test1 raster)

A faithful implementation of the current state-of-the-art retinal image-based
eye tracker — **strip-based registration to a composite/synthetic reference
frame** (the TSLO/AOSLO method of Stevenson & Roorda; the refined 2020 Roorda-lab
"robust strip-based digital image registration"; substrip variant Liu et al.
2024) — run on OUR real `test1` pursuit raster and scored with the EXACT same
`khz2d.evaluate` protocol as every other method (2 ms smoothing, 0.05 Hz drift
removal, per-axis affine calibration to the 0.2 Hz pursuit dot, r vs the ~32.5 Hz
machine tracker, precision = RMS of >25 ms detail).

## The SOTA algorithm (as implemented, `sota_strip.py`)

1. **Pre-processing.** Per-frame CLAHE contrast enhancement + de-band; blink /
   low-intensity frame rejection; distortion-frame rejection via the
   consecutive-frame full-frame match quality (`khz2d.chain` `q`) below 0.45.
2. **Composite (synthetic) reference.** Accepted low-distortion frames are
   averaged at their globally-registered positions into an oversized composite
   (a retinal mosaic) — registration is to this FIXED composite, NOT to the
   previous frame. This is the defining difference from incremental tracking.
   (632/1025 frames entered the composite; coverage 91%.)
3. **Strip registration.** Each frame is split into strips of S adjacent columns
   (a strip is parallel to the fast scanner = a column on this raster). Each
   strip is NCC-matched (`TM_CCOEFF_NORMED`) to the composite within a ±60 px
   local window; sub-pixel peak by 2D parabolic interpolation; accepted iff the
   NCC peak exceeds a threshold.
4. **High-rate trace.** Per-strip (x, y) offsets in temporal order give the eye
   trace at (808/S)·14.633 Hz: S=8 → 1478 Hz, S=1 → 11823 Hz (per column).

**Citations.** Stevenson & Roorda, Proc. SPIE 5688 (2005); Sheehy et al.,
Biomed. Opt. Express 3(10):2611 (2012); Bowers/Boehm/Roorda robust strip
registration (BOE, 2019–2020); Liu et al. substrip variant (2024).

## Implementation notes & honest simplifications

- **Global coordinate / reference selection.** The composite is built in the
  coordinate of `khz2d.chain()` — the robust incremental full-frame (strip-median)
  registration that EVERY method already uses as its 20 Hz absolute anchor. This
  replaces manual reference-frame selection. It is **not a handicap**: it gives
  the SOTA the *same* coarse anchor as M1/M4, so the comparison isolates the only
  thing under test — the high-rate per-strip residual estimator
  (composite-reference NCC vs previous-frame NCC vs particle filter).
- **Honest modality caveat.** The published sub-arcminute numbers are on
  cone-resolved AOSLO. On our video-rate NON-AO raster the composite is an
  averaged mosaic whose NCC peaks are intrinsically *lower* than a single sharp
  neighbour frame (median strip-NCC ≈ 0.31 to the composite). That lower SNR is
  exactly why "same data, same modality" is the only fair test — the SOTA is
  limited here by the same speckle/SNR as our methods.
- **Threshold / validity matching.** We report SOTA twice: at the *same numeric*
  strip-NCC acceptance as our incremental M1 (0.35), and at a *validity-matched*
  threshold (S=8 thr={thr8:.2f}; S=1 thr={thr1:.2f}) chosen so SOTA's valid% ≈
  the comparator's, so the precision number is computed on a comparable fraction
  of samples (a flat/over-rejected trace games precision — see guardrail below).
- **Compute.** S=8 over all 1025 frames (full 70 s). S=1 (per-column, 11823 Hz)
  over a 20 s window — the IDENTICAL window used for our published line-rate M4
  validation — because per-column NCC over the whole recording is heavy. Our M4
  numbers are reused from cache (the filter was not re-run).

## Guardrail (anti-gaming)

Precision is only meaningful if r_dot_x, r_trk_x and valid% are held. The
independent ~32.5 Hz tracker correlation `r_trk_x` is the honesty anchor;
`r_dot_x ≈ 0.9` is a pursuit-lag ceiling, not a target. We report all columns
together and do not crown a winner on precision alone.
"""


if __name__ == "__main__":
    main()
