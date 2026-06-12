"""make_method_overlays.py — fix the trajectory overlays.

The old `optimal_vs_previous_overlay.png` plotted the LINE-RATE runs (capped at
dur_s=20 s) over a 20-40 s window — where those runs have essentially no samples
and the dot's only motion (35-40 s) is past their coverage. It looked broken.

This regenerates trajectory overlays from FULL-LENGTH (70 s) caches over a
horizontally-active window (recon time ~4-14 s, the phase with the largest dot_x
excursion), for every method that makes sense:

  M0 chain (15 Hz) · M1 strips (1478 Hz) · M2 Kalman · M3 Viterbi ·
  M4 PF previous (physics) · M4 PF optimal (learned N=1000) · M5 MAP

Outputs:
  results/methods_overlay.png            — small-multiples, one panel per method
  results/optimal_vs_previous_overlay.png — fixed 3-config overlay (overwrite)
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import khz2d

RESULTS = khz2d.RESULTS
INK = "#1a1a1a"; MUTED = "#6b6b6b"; DOT = "#9a958a"
ACCENT = "#9c1f2e"; BLUE = "#22507a"; GOLD = "#c08a2e"; TEAL = "#2e7d74"
GREEN = "#4a7a3a"; PURPLE = "#5b4a8a"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.2,
})

# Full-length (70 s) methods that make sense to overlay, in order.
# (cache tag, label, colour)
METHODS = [
    ("m0_chain",                  "M0  frames only (15 Hz)",        MUTED),
    ("m1_s8",                     "M1  strips (1478 Hz)",           GREEN),
    ("m2_kalman",                 "M2  Kalman (11.8 kHz)",          GOLD),
    ("m3_viterbi",                "M3  Viterbi (11.8 kHz)",         PURPLE),
    ("m4_dpf_11823",              "M4  PF previous (11.8 kHz)",     BLUE),
    ("m4_dpf_1182_learned_n1000", "M4  PF optimal learned (1.2 kHz)", ACCENT),
    ("m5_map",                    "M5  MAP (11.8 kHz)",             TEAL),
]

T0, T1 = 4.0, 14.0          # horizontally-active window (recon time, s)
MAXPTS = 5000               # decimate dense traces for clarity/size


def _load_eval(tag, label):
    r = khz2d.load_method(tag)
    if r is None:
        return None
    ev = khz2d.evaluate(r["t"], r["x_px"], r["y_px"], r["valid"].astype(bool),
                        float(r["rate"]), label, smooth_ms=2)
    return ev


def _win(ev):
    t = ev["t"]
    m = (t >= T0) & (t <= T1)
    tw = t[m]
    cal = np.where(ev["valid"], ev["cal_x"], np.nan)[m]
    dot = ev["dot_x"][m]
    if tw.size > MAXPTS:                       # decimate for plotting
        s = max(1, tw.size // MAXPTS)
        tw, cal, dot = tw[::s], cal[::s], dot[::s]
    return tw, cal, dot


def small_multiples(evs, path=os.path.join(RESULTS, "methods_overlay.png")):
    n = len(evs)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.0 * nrow), sharex=True)
    axes = axes.ravel()
    for ax, (ev, (_t, label, c)) in zip(axes, evs):
        tw, cal, dot = _win(ev)
        ax.plot(tw, dot, color=DOT, lw=2.4, alpha=0.7, label="pursuit dot", zorder=1)
        ax.plot(tw, cal, color=c, lw=1.0, alpha=0.9, label="reconstruction", zorder=2)
        ax.set_title(f"{label}   ·   r={ev['r_dot_x']:.2f}"
                     + (f", prec={ev['prec_x']:.2f}'" if np.isfinite(ev['prec_x']) else ""),
                     loc="left")
        ax.set_ylabel("h. gaze (')")
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax in axes[max(0, n - ncol):n]:
        ax.set_xlabel("time (s)")
    handles = [plt.Line2D([0], [0], color=DOT, lw=2.4, label="pursuit dot (target)"),
               plt.Line2D([0], [0], color=INK, lw=1.2, label="method reconstruction")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.015))
    fig.suptitle(f"Calibrated horizontal gaze vs pursuit dot — all methods "
                 f"(test1, {T0:.0f}-{T1:.0f} s)", y=1.035, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fixed_optimal_overlay(path=os.path.join(RESULTS, "optimal_vs_previous_overlay.png")):
    """Overwrite the broken overlay: 3 configs, full-length 1182 Hz, active window."""
    cfg = [
        ("m4_dpf_1182",               "previous (physics N=300)", BLUE),
        ("m4_dpf_1182_learned_n1000", "optimal (learned N=1000)", ACCENT),
        ("m4_dpf_1182_n1000_b40",     "control (physics N=1000 B=40)", GOLD),
    ]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    drew_dot = False
    for tag, label, c in cfg:
        ev = _load_eval(tag, label)
        if ev is None:
            print("  MISSING", tag); continue
        tw, cal, dot = _win(ev)
        if not drew_dot:
            ax.plot(tw, dot, color=DOT, lw=2.6, alpha=0.7, label="pursuit dot (target)")
            drew_dot = True
        ax.plot(tw, cal, color=c, lw=1.0, alpha=0.9, label=label)
    ax.set_xlabel("time (s)"); ax.set_ylabel("horizontal gaze (arcmin)")
    ax.set_title(f"Calibrated horizontal trajectory vs pursuit dot "
                 f"(1182 Hz, full-length, {T0:.0f}-{T1:.0f} s)", pad=26)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.04))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    evs = []
    for tag, label, c in METHODS:
        ev = _load_eval(tag, label)
        if ev is None:
            print("  MISSING cache:", tag); continue
        evs.append((ev, (tag, label, c)))
    small_multiples(evs)
    fixed_optimal_overlay()
    print("done")
