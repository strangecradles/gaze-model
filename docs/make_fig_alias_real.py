"""make_fig_alias_real.py — EMPIRICAL aliasing figure for the project page.

Companion to the schematic ``fig_alias.png``: the same two-part story (the aliased
match score; the multimodal belief) but measured on REAL retinal lines, plus a
third panel that quantifies the payoff — what a per-line ``argmax`` estimator does
with the alias comb versus what the multimodal particle filter does, on the same
real data with known ground truth.

Why this is honest and not circular
------------------------------------
Everything is real retinal appearance at the NATIVE atlas scale (``col_step=1``):

* The reference is the registered multi-frame atlas ``A`` (``data.load_atlas`` —
  the averaged ``normal/`` SLO frames).
* Each observed line is a real measured line lifted from one of the individual
  ``normal/`` SLO frames. After registration, a feature at frame-``k`` row ``r``
  lands on atlas row ``r + dy_k`` (the stored per-frame offset aligns frame->ref),
  so the real line at atlas target ``(P*, A*)`` is ``frames[k][P*-dy_k, A*-dx_k:]``
  and its TRUE perp offset is ``P*`` by construction. We can therefore SCORE every
  estimator against a known truth without ever rendering the line we are trying to
  locate.
* The atlas average contains the source frame (1/20 leakage) — this makes the
  appearance match slightly easier for BOTH estimators equally, so the
  argmax-vs-filter comparison (the figure's claim) is unbiased.

Measured regime (see the probe in progress notes): real-line matching is genuinely
aliased even on the clean native atlas — peak-to-secondary ratio ~1.1, ~16-22
modes, lock strength (max NCC) ~0.4 — because a real line never matches the atlas
as well as a re-rendered atlas line would. The true peak only barely outranks its
alias rivals, so a per-line argmax lands on the wrong alias branch on a large
fraction of short, line-rate-length lines. That is the empirical motivation for
keeping the whole comb (the filter) instead of collapsing it (argmax).

Panels
------
(a) the alias comb of one representative real line: s(p) = NCC vs perp offset, with
    the true peak, the alias comb, the spacing Delta, and where the argmax lands.
(b) the multimodal belief that one line induces: a real bootstrap particle cloud
    (broad prior reweighted by this line's likelihood), coloured by weight, with the
    half-alias-spacing read-out window. The belief keeps every mode argmax discards.
(c) the payoff on a real known-truth sequence: per-line argmax error (which scatters
    onto the +-Delta, +-2Delta alias rails) versus the recursive filter's error
    (which stays on the true branch), as a time trace and as marginal histograms.

Run:  python docs/make_fig_alias_real.py
Writes docs/figures/fig_alias_real.png (220 dpi, house style from make_figures.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # headless; set before pyplot is imported anywhere
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

# repo root on the path so the data/method modules import (figure lives in docs/)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# shared house style (palette + rcParams + _save) — reused so this figure is
# visually identical in weight/font to the rest of the deck.
from make_figures import INK, MUTED, GRID, ACCENT, BLUE, _save  # noqa: E402

import calib            # noqa: E402
import data             # noqa: E402
import likelihood as lik  # noqa: E402
import filter as flt    # noqa: E402

DELTA = calib.ALIAS_SPACING_ROWS          # alias spacing in atlas rows (~124.6)
APR = calib.ARCMIN_PER_ROW                # arcmin per atlas row (~0.481)
DEG = DELTA * APR / 60.0                   # alias spacing in degrees (~1.0)

# Geometry of the demonstration (honest, line-rate-scale choices)
L = 128            # observed line length (samples). Short, as line-rate lines are.
RATE = 1500.0      # effective rate (Hz), above the f*~826 Hz aliasing gate
T = 1200           # sequence length (lines) ~ 0.8 s
N_PARTICLES = 400  # filter cloud size (paper default 300; 400 for a denser figure)


def rows_to_deg(rows):
    return np.asarray(rows, float) * APR / 60.0


# --------------------------------------------------------------------------- #
# Real known-truth line construction
# --------------------------------------------------------------------------- #
def real_line(atlas, k, P, A, length=L, col_step=1.0):
    """Real frame-``k`` line whose TRUE atlas perp is ``P`` (along start ``A``).

    Returns None if the window falls outside the frame (callers keep targets
    central so this never fires on the chosen trajectory).
    """
    H, W = atlas.ref_map.shape
    dy, dx = atlas.offsets[k]
    r = int(round(P - dy))
    cols = (A - dx + col_step * np.arange(length)).astype(int)
    if r < 0 or r >= H or cols.min() < 0 or cols.max() >= W:
        return None
    return atlas.frames[k][r, cols].astype(np.float64)


def make_trajectory(seed=1):
    """A known, smooth perp/along gaze path (drift + a few microsaccades).

    Kept within a central, always-in-FOV box so the L-sample window is valid for
    every frame's registration offset. The trace is only mildly consistent with
    the filter's OU+main-sequence prior (drift + occasional steps); it is NOT
    generated by the decoder, so the appearance comes purely from real frames.
    """
    rng = np.random.default_rng(seed)
    P0, A0 = 300.0, 560.0

    def smooth_drift(amp_rows, tau_s):
        x = np.cumsum(rng.normal(0.0, 1.0, T))
        x = gaussian_filter1d(x, RATE * tau_s)
        x = (x - x.mean()) / (x.std() + 1e-9)
        return x * amp_rows

    perp = P0 + smooth_drift(0.30 * DELTA, 0.10)
    along = A0 + smooth_drift(60.0, 0.12)

    # a few straight microsaccades: coupled perp/along steps that persist
    n_sac = 3
    t_sac = np.sort(rng.choice(np.arange(int(0.1 * T), int(0.9 * T)), n_sac,
                               replace=False))
    for ts in t_sac:
        dp = rng.choice([-1.0, 1.0]) * rng.uniform(0.30, 0.50) * DELTA
        da = rng.choice([-1.0, 1.0]) * rng.uniform(30.0, 70.0)
        ramp = int(RATE * 0.008)  # ~8 ms ballistic ramp
        prof = np.clip((np.arange(T) - ts) / max(ramp, 1), 0.0, 1.0)
        perp = perp + dp * prof
        along = along + da * prof

    perp = np.clip(perp, 205.0, 405.0)
    along = np.clip(along, 440.0, 705.0)
    return perp.astype(np.float64), along.astype(np.float64)


# --------------------------------------------------------------------------- #
# Build the real sequence; run argmax-per-line and the recursive filter
# --------------------------------------------------------------------------- #
def build_data():
    atlas = data.load_atlas()
    n_frames = atlas.frames.shape[0]
    perp_true, along_true = make_trajectory()

    lines = np.empty((T, L), np.float64)
    for t in range(T):
        k = t % n_frames
        ln = real_line(atlas, k, perp_true[t], along_true[t])
        if ln is None:  # should not happen for the clamped trajectory
            ln = np.zeros(L)
        lines[t] = ln

    # (1) per-line argmax of the fine-band perp likelihood (the single-hypothesis
    #     point estimator the paper contrasts against)
    argmax_perp = np.empty(T)
    psr = np.empty(T)
    nmodes = np.empty(T)
    maxncc = np.empty(T)
    rep = {}  # cache the representative line's full curve
    for t in range(T):
        rows, sc = lik.perp_likelihood(lines[t], atlas, float(along_true[t]),
                                       band="fine", col_step=1.0)
        argmax_perp[t] = rows[int(np.argmax(sc))]
        psr[t] = lik.psr(sc)
        nmodes[t] = lik.n_modes(sc)
        maxncc[t] = float(sc.max())
        rep[t] = (rows, sc)

    # (2) the recursive multimodal particle filter on the SAME lines. It additionally
    #     uses the trusted along measurement and a broad coarse perp anchor (truth +
    #     ~43' noise) — exactly the architecture of Section 4; this is the honest
    #     "filter vs per-line argmax" comparison.
    rng = np.random.default_rng(0)
    coarse_anchor = perp_true + rng.normal(0.0, flt.COARSE_SIGMA_ROWS, T)
    along_meas = along_true + rng.normal(0.0, 1.0, T)
    res = flt.run(
        lines, along_meas, RATE, atlas,
        init_perp=float(perp_true[0]), init_along=float(along_true[0]),
        n_particles=N_PARTICLES, perp_spread=DELTA, along_spread=3.0,
        line_len=L, col_step=1.0, coarse_anchor=coarse_anchor, seed=0,
    )

    err_argmax = (argmax_perp - perp_true)
    err_filter = (res.est_perp - perp_true)
    return dict(atlas=atlas, perp_true=perp_true, along_true=along_true,
                lines=lines, rep=rep, argmax_perp=argmax_perp, psr=psr,
                nmodes=nmodes, maxncc=maxncc, res=res,
                err_argmax=err_argmax, err_filter=err_filter)


def pick_representative(D):
    """The clearest teaching case: a line where the per-line argmax mislocks by
    ~one alias spacing AND the true peak survives as a near-equal rival (so panel
    (a) shows a near-tie that argmax loses, and panel (b) visibly keeps the true
    mode). Among such lines we maximise the true-peak score relative to the global
    max."""
    half = 0.5 * DELTA
    best = None
    for t in range(T):
        err = D["argmax_perp"][t] - D["perp_true"][t]
        if abs(err) <= half:                          # require a genuine mislock
            continue
        if abs(abs(err) - DELTA) > 0.28 * DELTA:       # ~ one alias over
            continue
        rows, sc = D["rep"][t]
        near0 = np.where(np.abs(rows - D["perp_true"][t]) <= 12)[0]
        true_ratio = float(sc[near0].max() / (sc.max() + 1e-9))
        if best is None or true_ratio > best[0]:
            best = (true_ratio, t)
    return best[1] if best else int(np.argmax(np.abs(D["err_argmax"])))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(D):
    t0 = pick_representative(D)
    rows, sc = D["rep"][t0]
    atlas = D["atlas"]
    Ptrue = D["perp_true"][t0]
    Atrue = D["along_true"][t0]
    off_deg = rows_to_deg(rows - Ptrue)
    am_row = D["argmax_perp"][t0]
    am_off = rows_to_deg(am_row - Ptrue)

    # population alias statistics (for honest captions / annotations)
    half = 0.5 * DELTA
    mis_argmax = float(np.mean(np.abs(D["err_argmax"]) > half) * 100)
    mis_filter = float(np.mean(np.abs(D["err_filter"]) > half) * 100)
    rms_argmax = float(np.sqrt(np.mean((D["err_argmax"] * APR) ** 2)))
    rms_filter = float(np.sqrt(np.mean((D["err_filter"] * APR) ** 2)))
    med_psr = float(np.median(D["psr"]))
    med_nm = float(np.median(D["nmodes"]))
    med_ncc = float(np.median(D["maxncc"]))

    fig = plt.figure(figsize=(7.4, 5.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05],
                  hspace=0.62, wspace=0.28,
                  left=0.085, right=0.975, top=0.92, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    # bottom row: a wide time trace + a slim shared-y marginal histogram
    gs_c = gs[1, :].subgridspec(1, 2, width_ratios=[4.4, 1.0], wspace=0.04)
    ax_c = fig.add_subplot(gs_c[0, 0])
    ax_h = fig.add_subplot(gs_c[0, 1], sharey=ax_c)

    XLIM = (-2.6, 2.6)
    RAILS = (-2.0, -1.0, 1.0, 2.0)   # alias rails at integer multiples of Delta (~1 deg)

    # ----- (a) the alias comb of one real line -----
    # The raw fine-band NCC over candidate rows is genuinely spiky; we draw it faint
    # and overlay a light envelope (sigma 1 row, which preserves the sharp true and
    # alias peaks) so the comb is legible. The estimator acts on the raw curve.
    sc_s = gaussian_filter1d(sc, 1.0)
    for rr in RAILS:
        ax_a.axvline(rr, color=GRID, lw=0.9, zorder=0)
    ax_a.fill_between(off_deg, 0, sc_s, color=BLUE, alpha=0.10, lw=0, zorder=1)
    ax_a.plot(off_deg, sc, color=BLUE, lw=0.5, alpha=0.35, zorder=2)
    ax_a.plot(off_deg, sc_s, color=BLUE, lw=1.5, zorder=3)
    ymax = max(0.62, sc.max() * 1.26)
    for p in lik.top_peaks(sc_s, k=9, distance=int(0.22 * DELTA)):
        o = off_deg[p]
        if abs(o) < 0.10 or not (XLIM[0] < o < XLIM[1]):
            continue
        ax_a.plot([o], [sc_s[p]], marker="v", ms=3.4, color=MUTED, zorder=4)
    # the true peak (a strong, sharp secondary here — argmax just missed it)
    near0 = np.where(np.abs(rows - Ptrue) <= 12)[0]
    i_true = near0[int(np.argmax(sc[near0]))]
    ax_a.plot([off_deg[i_true]], [sc[i_true]], marker="o", ms=4.6, color=ACCENT,
              zorder=5)
    ax_a.axvline(0.0, color=ACCENT, lw=1.1, ls=(0, (3, 2)), zorder=3)
    ax_a.text(0.0, ymax * 0.95, "true", color=ACCENT, ha="center", fontsize=7.2)
    # the argmax pick (here it mislocks onto the +1 alias)
    ax_a.plot([am_off], [sc[int(np.argmax(sc))]], marker="o", ms=6.5,
              mfc="none", mec=INK, mew=1.5, zorder=6)
    lx = float(np.clip(am_off + (0.45 if am_off >= 0 else -0.45), -2.3, 2.3))
    ax_a.annotate("argmax\n(mislock)", xy=(am_off, sc.max()),
                  xytext=(lx, min(sc.max() * 1.32, ymax * 0.95)),
                  fontsize=6.6, color=INK, ha="center",
                  arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    ax_a.annotate("", xy=(DEG, 0.10), xytext=(0.0, 0.10),
                  arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ax_a.text(DEG / 2, 0.115, r"$\Delta\!\approx\!1^\circ$", ha="center",
              va="bottom", fontsize=6.6, color=INK)
    ax_a.set_xlim(*XLIM)
    ax_a.set_ylim(0, ymax)
    ax_a.set_xlabel("perp offset  (deg)")
    ax_a.set_ylabel("match score  (NCC)")
    ax_a.set_title("(a)  one real line is aliased", loc="left")
    ax_a.text(0.975, 0.95,
              f"max NCC {sc.max():.2f}\nPSR {lik.psr(sc):.2f}\n{lik.n_modes(sc)} modes",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=6.3,
              color=MUTED, linespacing=1.3)

    # ----- (b) the multimodal belief the SAME line induces -----
    # The per-line observation belief is the filter's appearance weight with a flat
    # prior: b(p) = exp(beta*(s(p) - max s)) (filter.BETA). It is the normalized comb
    # of (a): argmax collapses it to the single +1 peak, but the belief keeps the
    # true mode (0) as a near-equal rival. We draw the belief density + a particle
    # rug sampled from it (coloured by weight).
    belief = np.exp(flt.BETA * (sc - sc.max()))
    bden = gaussian_filter1d(belief, 1.0)
    bden = bden / (bden.max() + 1e-12)
    base, ht = 0.16, 0.80
    ax_b.fill_between(off_deg, base, base + ht * bden, color=BLUE, alpha=0.16, lw=0)
    ax_b.plot(off_deg, base + ht * bden, color=BLUE, lw=1.5)
    # particle rug: resample particle offsets proportional to the belief
    win = (off_deg > XLIM[0]) & (off_deg < XLIM[1])
    pmf = belief[win] / belief[win].sum()
    rngb = np.random.default_rng(5)
    samp = rngb.choice(off_deg[win], size=1400, p=pmf)
    samp = samp + rngb.normal(0.0, 0.012, samp.size)        # tiny jitter
    wcol = np.interp(samp, off_deg, belief)
    ax_b.scatter(samp, rngb.uniform(0.015, 0.13, samp.size), s=4.5, c=wcol,
                 cmap="viridis", alpha=0.65, edgecolors="none", vmin=0, zorder=2)
    ax_b.axvline(0.0, color=ACCENT, lw=1.1, ls=(0, (3, 2)))
    ax_b.text(0.0, 1.16, "true", color=ACCENT, ha="center", fontsize=7.2)
    side = -1.0 if am_off < 0 else 1.0
    # the true mode the belief keeps (argmax discarded it)
    ax_b.plot([0.0], [base + ht * bden[i_true]], marker="o", ms=4.6, color=ACCENT,
              zorder=4)
    ax_b.annotate("true mode\nkept", xy=(0.0, base + ht * bden[i_true]),
                  xytext=(-side * 1.25, 1.0), fontsize=6.4, color=INK, ha="center",
                  arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    # the single peak argmax keeps (same symbol as panel a)
    i_am = int(np.argmax(sc))
    ax_b.plot([am_off], [base + ht * bden[i_am]], marker="o", ms=6.5, mfc="none",
              mec=INK, mew=1.5, zorder=5)
    ax_b.annotate("argmax keeps\nonly this peak", xy=(am_off, base + ht * bden[i_am]),
                  xytext=(float(np.clip(am_off + side * 0.95, -2.3, 2.3)), 1.13),
                  fontsize=6.3, color=INK, ha="center",
                  arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    ax_b.set_xlim(*XLIM)
    ax_b.set_ylim(0, 1.30)
    ax_b.set_yticks([])
    ax_b.set_xlabel("perp offset  (deg)")
    ax_b.set_ylabel("belief density")
    ax_b.set_title("(b)  the belief keeps every mode", loc="left")

    # ----- (c) the payoff: argmax vs filter on a real known-truth sequence -----
    tms = np.arange(T) / RATE * 1e3
    ea = D["err_argmax"] * APR / 60.0  # deg
    ef = D["err_filter"] * APR / 60.0
    for kk in (1, 2):
        for s in (1, -1):
            ax_c.axhline(s * kk * DEG, color=MUTED, lw=0.6, ls=(0, (1, 2)),
                         zorder=0)
            ax_c.text(tms[-1] * 0.997, s * kk * DEG, f"{s*kk:+d}$\\Delta$",
                      fontsize=5.6, color=MUTED, ha="right", va="center")
    ax_c.axhline(0.0, color=ACCENT, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax_c.scatter(tms, ea, s=3.6, color=ACCENT, alpha=0.32, edgecolors="none",
                 zorder=2, label="per-line argmax")
    ax_c.plot(tms, ef, color=BLUE, lw=1.1, zorder=3, label="particle filter")
    ax_c.set_xlim(0, tms[-1])
    ax_c.set_ylim(-2.7, 2.7)
    ax_c.set_xlabel("time  (ms)")
    ax_c.set_ylabel("perp error  (deg)")
    ax_c.set_title("(c)  real sequence, known truth", loc="left")
    ax_c.text(0.012, 0.05,
              f"argmax:  RMS {rms_argmax:.0f}$'$   ·   {mis_argmax:.0f}% on an alias\n"
              f"filter:    RMS {rms_filter:.0f}$'$   ·   {mis_filter:.0f}% on an alias",
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=6.6,
              color=INK, linespacing=1.5,
              bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.6))
    ax_c.legend(loc="upper left", fontsize=6.8, ncol=2, handletextpad=0.4,
                columnspacing=1.0, bbox_to_anchor=(0.0, 1.0))

    # marginal error histograms (shared y) — the empirical "alias structure"
    bins = np.linspace(-2.7, 2.7, 61)
    ax_h.hist(ea, bins=bins, orientation="horizontal", color=ACCENT, alpha=0.5,
              density=True)
    ax_h.hist(ef, bins=bins, orientation="horizontal", color=BLUE, alpha=0.6,
              density=True)
    for kk in (1, 2):
        for s in (1, -1):
            ax_h.axhline(s * kk * DEG, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax_h.set_xticks([])
    ax_h.tick_params(labelleft=False)
    ax_h.set_xlabel("error\ndensity", fontsize=6.6)
    for sp in ("top", "right", "bottom"):
        ax_h.spines[sp].set_visible(False)

    _save(fig, "fig_alias_real.png")

    # diagnostics so the captions/prose use measured numbers
    print("\n=== empirical alias figure — measured numbers ===")
    print(f"Delta = {DELTA:.2f} rows = {DEG:.3f} deg = {DELTA*APR:.1f}'")
    print(f"representative line t0={t0}: argmax err {am_off:+.2f} deg "
          f"({(am_row-Ptrue)*APR:+.1f}'), maxNCC {sc.max():.3f}, "
          f"PSR {lik.psr(sc):.2f}, n_modes {lik.n_modes(sc)}")
    print(f"sequence medians: maxNCC {med_ncc:.3f}, PSR {med_psr:.2f}, "
          f"n_modes {med_nm:.1f}")
    print(f"argmax : RMS {rms_argmax:.2f}'  mislock {mis_argmax:.1f}%")
    print(f"filter : RMS {rms_filter:.2f}'  mislock {mis_filter:.1f}%")
    print(f"argmax errors within 0.25Δ of ±1Δ: "
          f"{np.mean(np.abs(np.abs(D['err_argmax'])-DELTA)<0.25*DELTA)*100:.1f}%, "
          f"of ±2Δ: "
          f"{np.mean(np.abs(np.abs(D['err_argmax'])-2*DELTA)<0.25*DELTA)*100:.1f}%")


if __name__ == "__main__":
    D = build_data()
    make_figure(D)
