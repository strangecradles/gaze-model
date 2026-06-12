"""make_figures.py — publication-quality figures for the project page.

Generates four vector-crisp PNGs into docs/figures/ with a single shared style so
fonts and line weights are consistent across panels (the cardinal rule for
professional figures):

  fig_system.png    — the DPF architecture / dataflow diagram (schematic)
  fig_alias.png     — the aliasing problem and the multimodal belief (schematic)
  fig_ratesweep.png — accuracy + gross-error persistence vs rate (real G13 numbers)
  fig_methods.png   — per-line precision by method (real testbed-A numbers)

The two data figures are re-rendered from the numbers in results/*.md; the two
schematics are explicitly conceptual (labelled as such in the captions).

Run:  python docs/make_figures.py
"""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import os

# --------------------------------------------------------------------------- #
# Shared style — a restrained, modern academic look (sans labels, thin spines).
# --------------------------------------------------------------------------- #
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e6e3da"
ACCENT = "#9c1f2e"     # oxblood (primary)
BLUE = "#22507a"       # deep blue (secondary)
GOLD = "#c08a2e"       # amber (tertiary)
TEAL = "#2e7d74"

mpl.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "text.color": INK,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------- #
# Figure 1 — System / dataflow diagram
# --------------------------------------------------------------------------- #
def fig_system():
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, tc=INK, fs=8.2, weight="normal", r=0.025):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.4,rounding_size={r*100}",
                           linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, weight=weight, zorder=3, linespacing=1.3)

    def arrow(x0, y0, x1, y1, color=INK, ls="-", lw=1.2, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle="-|>", mutation_scale=11, lw=lw, color=color,
                     linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1))

    # --- Inputs (left column), each aligned with the stage it feeds ---
    box(1, 26.5, 20, 6.2, "SLO frames\n= atlas (15 Hz)", "#f7eef0", ACCENT, fs=8.0)
    box(1, 18.5, 20, 6.2, "Fast line $z_t$\n(11.8 kHz)", "#eef2f6", BLUE, fs=8.0)
    box(1, 10.5, 20, 6.2, "Slow 2-D anchor\n(registration)", "#fbf3e6", GOLD, fs=7.7)

    # --- Core recursive block (center) ---
    cx, cy, cw, ch = 27, 3, 46, 44
    panel = FancyBboxPatch((cx, cy), cw, ch,
                           boxstyle="round,pad=0.4,rounding_size=2.0",
                           linewidth=1.1, edgecolor=INK, facecolor="#fcfcfb",
                           zorder=1)
    ax.add_patch(panel)
    ax.text(cx + cw / 2, cy + ch - 2.6, "Particle filter  (one step per line)",
            ha="center", va="center", fontsize=8.6, weight="bold", color=INK)

    bx = cx + 3.5
    bw = cw - 7
    box(bx, 34.5, bw, 6.2, "1.  Predict — IMM prior\npursuit OU  /  saccade main sequence",
        "#eef2f6", BLUE, fs=7.7)
    box(bx, 26.5, bw, 6.2, "2.  Render — frozen decoder\n$\\hat z(x_i)=\\mathrm{render}(p_i,a_i;\\,\\mathrm{atlas})$",
        "#f4f1ea", MUTED, fs=7.7)
    box(bx, 18.5, bw, 6.2, "3.  Weight — multimodal\n$w_i\\propto e^{\\beta\\,\\mathrm{NCC}_i}\\cdot w^{\\mathrm{along}}_i\\cdot w^{\\mathrm{couple}}_i$",
        "#f7eef0", ACCENT, fs=7.7)
    box(bx, 10.5, bw, 6.2, "4.  Estimate  +  resample / reseed\nalias-robust mean · ESS · NCC gate",
        "#eef5f1", TEAL, fs=7.7)

    # vertical flow inside the panel
    for y0, y1 in [(34.5, 32.7), (26.5, 24.7), (18.5, 16.7)]:
        arrow(cx + cw / 2, y0, cx + cw / 2, y1, color=MUTED, lw=1.1)

    # recurrence: estimate -> predict (state feedback), arc on the right edge
    arrow(bx + bw + 1.5, 13.9, bx + bw + 1.5, 37.3, color=BLUE,
          ls=(0, (4, 2)), lw=1.1, rad=-0.12)
    ax.text(bx + bw + 3.0, 25.6, "state $x_{t-1}\\!\\to\\!x_t$", rotation=90,
            ha="center", va="center", fontsize=6.6, color=BLUE)

    # --- Output (right) ---
    box(79, 21.5, 20, 9, "Gaze estimate\n$(\\hat p_t,\\hat a_t)$\nat line rate", "#eef5f1", TEAL, fs=8.2)

    # --- input -> core arrows (horizontal, semantically wired) ---
    arrow(21.4, 29.6, bx - 0.6, 29.6, color=ACCENT)   # frames=atlas -> render (2)
    arrow(21.4, 21.6, bx - 0.6, 21.6, color=BLUE)     # observation z_t -> weight (3)
    arrow(21.4, 13.6, bx - 0.6, 13.6, color=GOLD)     # anchor -> reseed (4)
    # core -> output
    arrow(cx + cw, 25, 79 - 0.5, 26, color=TEAL)

    _save(fig, "fig_system.png")


# --------------------------------------------------------------------------- #
# Figure 2 — Aliasing & the multimodal belief (schematic)
# --------------------------------------------------------------------------- #
def fig_alias():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), width_ratios=[1, 1.15])

    # (a) a single line matches at several offsets -> multimodal score
    ax = axes[0]
    x = np.linspace(-2.6, 2.6, 1000)
    alias = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    amp = np.array([0.45, 0.7, 1.0, 0.62, 0.4])
    width = 0.085
    y = sum(a * np.exp(-0.5 * ((x - m) / width) ** 2) for a, m in zip(amp, alias))
    ax.plot(x, y, color=BLUE, lw=1.7)
    ax.fill_between(x, 0, y, color=BLUE, alpha=0.08)
    ax.axvline(0.0, color=ACCENT, lw=1.1, ls=(0, (3, 2)))
    ax.text(0.0, 1.08, "true", color=ACCENT, ha="center", fontsize=7.5)
    for m in alias[alias != 0]:
        ax.text(m, amp[list(alias).index(m)] + 0.05, "alias", color=MUTED,
                ha="center", fontsize=6.6)
    ax.annotate("", xy=(1.0, 0.30), xytext=(0.0, 0.30),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ax.text(0.5, 0.36, "1 alias\nspacing", ha="center", va="bottom", fontsize=6.6,
            color=INK)
    ax.set_xlabel("perp offset  (deg)")
    ax.set_ylabel("match score  (NCC)")
    ax.set_title("(a)  one line is aliased", loc="left")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1.25)

    # (b) particle cloud spreads over the peaks; estimate from the dominant mode
    ax = axes[1]
    rng = np.random.default_rng(3)
    centers = np.array([-1.0, 0.0, 1.0, 2.0])
    weights = np.array([0.18, 0.55, 0.19, 0.08])
    pts = []
    wts = []
    for c, w in zip(centers, weights):
        n = int(300 * w)
        s = rng.normal(c, 0.07, n)
        pts.append(s)
        wts.append(np.full(n, w))
    pts = np.concatenate(pts)
    wts = np.concatenate(wts)
    jitter = rng.uniform(0, 1, pts.size)
    ax.scatter(pts, jitter, s=6, c=wts, cmap="viridis", alpha=0.7, edgecolors="none")
    ax.axvspan(-0.5, 0.5, color=ACCENT, alpha=0.06)
    ax.axvline(0.0, color=ACCENT, lw=1.1, ls=(0, (3, 2)))
    ax.text(0.0, 1.12, "alias-robust\nestimate", color=ACCENT, ha="center",
            fontsize=6.8)
    ax.set_xlabel("perp offset  (deg)")
    ax.set_yticks([])
    ax.set_xlim(-2.0, 2.6)
    ax.set_ylim(0, 1.35)
    ax.set_title("(b)  the cloud keeps every mode", loc="left")

    fig.tight_layout()
    _save(fig, "fig_alias.png")


# --------------------------------------------------------------------------- #
# Figure 3 — Rate sweep (real numbers, results/rate_sweep_verdict.md, G13)
# --------------------------------------------------------------------------- #
def fig_ratesweep():
    rate = np.array([60, 344, 820, 1500, 2000, 4000, 12000], float)
    fix_rms = np.array([54.22, 12.28, 6.85, 1.57, 1.09, 1.70, 0.86])
    sac_rms = np.array([121.2, 62.9, 51.7, 35.9, 53.1, 46.7, 38.4])
    pers = np.array([500.0, 26.2, 15.9, 10.0, 4.5, 4.8, 1.7])
    GATE = 826.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    ax.axvspan(40, GATE, color=GRID, alpha=0.6, lw=0)
    ax.axhline(6.0, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    ax.text(70, 6.6, "0.1° (sub-cone) target", fontsize=6.8, color=MUTED)
    ax.plot(rate, fix_rms, "-o", color=BLUE, label="fixation / pursuit")
    ax.plot(rate, sac_rms, "-s", color=ACCENT, label="through-saccade",
            markerfacecolor="white")
    ax.axvline(GATE, color=INK, lw=0.8, ls=":")
    ax.text(GATE * 0.92, 1.05, "alias gate $V_{\\max}/\\Delta$", fontsize=6.8,
            color=INK, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("effective rate  (Hz)")
    ax.set_ylabel("perp RMS error  (arcmin)")
    ax.set_title("(a)  accuracy vs rate", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.axvspan(40, GATE, color=GRID, alpha=0.6, lw=0)
    ax.plot(rate, pers, "-o", color=GOLD)
    ax.axvline(GATE, color=INK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("effective rate  (Hz)")
    ax.set_ylabel("worst gross-error run  (ms)")
    ax.set_title("(b)  mislock persistence collapses", loc="left")
    ax.annotate("500 ms", xy=(60, 500), xytext=(120, 360),
                fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax.annotate("1.7 ms", xy=(12000, 1.7), xytext=(2600, 2.4),
                fontsize=6.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))

    fig.tight_layout()
    _save(fig, "fig_ratesweep.png")


# --------------------------------------------------------------------------- #
# Figure 4 — Per-line precision by method (real numbers, results/khz2d_methods.md)
# --------------------------------------------------------------------------- #
def fig_methods():
    # horizontal (aliased axis) detail precision, arcmin — lower is better.
    methods = ["M1 strips", "M2 Kalman", "M3 Viterbi", "M5 MAP",
               "M4 PF (learned)", "M4 PF"]
    prec = [4.39, 3.20, 2.87, 2.88, 1.88, 1.76]
    colors = [MUTED, MUTED, MUTED, MUTED, BLUE, ACCENT]

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    y = np.arange(len(methods))
    ax.barh(y, prec, color=colors, height=0.62, zorder=3)
    for yi, v in zip(y, prec):
        ax.text(v + 0.06, yi, f"{v:.2f}'", va="center", fontsize=7.6,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("horizontal detail precision  (arcmin,  lower = better)")
    ax.set_xlim(0, 5.1)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Fine-motion precision on the aliased axis  (real raster, 11.8 kHz)",
                 loc="left")
    leg = [Line2D([0], [0], color=ACCENT, lw=6, label="proposed (physics)"),
           Line2D([0], [0], color=BLUE, lw=6, label="proposed (learned)"),
           Line2D([0], [0], color=MUTED, lw=6, label="baselines")]
    ax.legend(handles=leg, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_methods.png")


# --------------------------------------------------------------------------- #
# Figure 5 — Design space: each pipeline stage and its candidate upgrades
# --------------------------------------------------------------------------- #
def fig_agenda():
    stages = [
        ("Dynamics prior",
         "IMM: pursuit OU\n+ saccade main seq.",
         [("Neural ODE / deep\nMarkov model", "ml"),
          ("Learned generative\noculomotor prior", "ml"),
          ("Diffusion trajectory\nprior (denoiser)", "ml"),
          ("Richer hand model\n(tremor, PSO)", "cl")]),
        ("Proposal",
         "Bootstrap\n(prior = proposal)",
         [("Cond. normalizing-\nflow proposal", "ml"),
          ("Observation-aware\namortized proposal", "ml"),
          ("Guided / twisted\nproposal", "cl")]),
        ("Observation\nlikelihood",
         "Fine-band NCC\n(physics render)",
         [("Learned calibrated\nlikelihood head", "ml"),
          ("Splatting / neural\nfield decoder", "ml"),
          ("Self-supervised\nfeature match", "ml"),
          ("Blur-forward\nmodel", "cl")]),
        ("Resampling",
         "Systematic +\nroughening",
         [("Entropy-OT\n(Sinkhorn)", "ml"),
          ("Stop-gradient\nreparam.", "ml"),
          ("Soft / learned\nresampler", "ml")]),
        ("Inference &\ntraining",
         "Hand-tuned,\nmodular",
         [("End-to-end DPF\n(FIVO / VSMC)", "ml"),
          ("Score-based filter\n/ assimilation", "ml"),
          ("Amortized /\ntransformer filter", "ml")]),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(14, 100)
    ax.axis("off")

    n = len(stages)
    col_w = 100.0 / n
    pad = 1.6
    ML = "#eef2f6"; ML_E = BLUE
    CL = "#f4f1ea"; CL_E = MUTED

    for i, (title, current, ups) in enumerate(stages):
        x0 = i * col_w + pad
        w = col_w - 2 * pad
        cx = x0 + w / 2
        # stage title
        ax.text(cx, 96, title, ha="center", va="center", fontsize=8.4,
                weight="bold", color=INK, linespacing=1.15)
        # current choice (oxblood, the established baseline)
        p = FancyBboxPatch((x0, 80), w, 9,
                           boxstyle="round,pad=0.3,rounding_size=1.2",
                           linewidth=1.1, edgecolor=ACCENT, facecolor="#f7eef0",
                           zorder=2)
        ax.add_patch(p)
        ax.text(cx, 84.5, current, ha="center", va="center", fontsize=6.7,
                color=INK, linespacing=1.15)
        # downward arrow + label beside it
        ax.add_patch(FancyArrowPatch((cx, 79.4), (cx, 73.0), arrowstyle="-|>",
                     mutation_scale=9, lw=1.0, color=MUTED, zorder=1))
        ax.text(cx, 75.8, "upgrades", ha="center", va="center",
                fontsize=5.8, style="italic", color=MUTED,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
        # candidate upgrades stacked
        y = 70.5
        bh = 8.2
        gap = 1.7
        for label, kind in ups:
            fc, ec = (ML, ML_E) if kind == "ml" else (CL, CL_E)
            p = FancyBboxPatch((x0, y - bh), w, bh,
                               boxstyle="round,pad=0.3,rounding_size=1.0",
                               linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2)
            ax.add_patch(p)
            ax.text(cx, y - bh / 2, label, ha="center", va="center",
                    fontsize=6.3, color=INK, linespacing=1.12)
            y -= (bh + gap)

    # legend
    leg = [Line2D([0], [0], marker="s", color="none", markerfacecolor="#f7eef0",
                  markeredgecolor=ACCENT, markersize=11, label="current baseline"),
           Line2D([0], [0], marker="s", color="none", markerfacecolor=ML,
                  markeredgecolor=BLUE, markersize=11, label="ML / learned lever"),
           Line2D([0], [0], marker="s", color="none", markerfacecolor=CL,
                  markeredgecolor=MUTED, markersize=11, label="classical lever")]
    ax.legend(handles=leg, loc="lower center", ncol=3, fontsize=7,
              bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.4)
    _save(fig, "fig_agenda.png")


# --------------------------------------------------------------------------- #
# Figure 6 — Head-to-head vs the SOTA strip-registration baseline (real test1)
#   numbers from results/sota_comparison.md + results/real_eye_optimization.md
# --------------------------------------------------------------------------- #
def fig_sota():
    # (label, prec_x arcmin, colour)   — lower is better
    tiers = [
        ("(a)  ~1.5 kHz tier", [
            ("SOTA strip\n(composite ref)", 3.00, MUTED),
            ("strip\n(incremental)", 4.39, "#b7b2a6"),
            ("ours: physics\nPF", 2.05, BLUE),
            ("ours: best\nPF", 1.90, ACCENT),
        ]),
        ("(b)  11.8 kHz line rate", [
            ("SOTA strip\n(composite ref)", 4.38, MUTED),
            ("strip\n(incremental)", 5.94, "#b7b2a6"),
            ("ours: physics\nPF", 1.66, BLUE),
            ("ours: best\nPF", 1.57, ACCENT),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, (title, rows) in zip(axes, tiers):
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        cols = [r[2] for r in rows]
        x = np.arange(len(rows))
        ax.bar(x, vals, color=cols, width=0.66, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.08, f"{v:.2f}'", ha="center", va="bottom",
                    fontsize=7.6, color=INK)
        # SOTA reference line + improvement arrow to best PF
        sota = vals[0]; best = vals[-1]
        ax.axhline(sota, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        pct = 100.0 * (sota - best) / sota
        ax.annotate(f"{pct:.0f}% better", xy=(len(rows) - 1, best),
                    xytext=(len(rows) - 1, sota), ha="center", va="bottom",
                    fontsize=7.2, color=ACCENT,
                    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.9))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("horizontal precision (arcmin, lower = better)")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Per-line precision on real test1: our particle filter vs the SOTA "
                 "strip-registration baseline", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig_sota.png")


if __name__ == "__main__":
    fig_system()
    fig_alias()
    fig_ratesweep()
    fig_methods()
    fig_agenda()
    fig_sota()
    print("all figures written to", OUT)
