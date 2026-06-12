"""optimize_real_eye.py — search→run→evaluate→refine loop to optimize the M4
particle-filter gaze reconstruction on the REAL human ``test1`` raster.

GOAL: beat the current best HORIZONTAL PRECISION (RMS of >25 ms detail, the kHz
payoff metric; lower is better) without degrading the honesty anchors.

Current SOTA on real test1 (from results/optimal_vs_previous_test1.md):
  previous physics N=300 : prec_x 1.66' @11823, 2.05' @1182
  optimal  learned N=1000: prec_x 1.62' @11823, 1.98' @1182
  honesty anchors @1182  : r_dot_x 0.901/0.905, r_trk_x 0.550/0.551, valid 70/71%

METHODOLOGICAL GUARDRAIL (do NOT game the metric): precision can be lowered
trivially by over-smoothing / lazy tracking (a flat line has ~0 precision AND
~0 correlation). A precision win is REAL only if it does NOT degrade r_dot_x,
r_trk_x, or valid%. Success = lower prec_x while holding
  r_dot_x >= base_r_dot - 0.005,  r_trk_x >= base_r_trk - 0.005,  valid >= base_valid - 2pts.
r_trk_x (independent ~32.5 Hz machine tracker) is the honesty anchor; r_dot_x is
a pursuit-lag CEILING (the dot is the target, not the eye), not a target to max.

COMPUTE STRATEGY: broad search at 1182 Hz full-length (cheap, ~1 min/config for
physics N=300); validate top configs at the 11823 Hz line rate over a matched
dur_s window. Everything caches under cache/ with descriptive tags via m4_dpf.

Run:
  python optimize_real_eye.py --phase broad      # 1182 full-length sweep
  python optimize_real_eye.py --phase validate   # 11823 dur-matched top configs
  python optimize_real_eye.py --phase all
"""
from __future__ import annotations

import argparse
import time
import numpy as np

import khz2d
import khz2d_methods as M


# ---------------------------------------------------------------------------
# Baselines (the SOTA to beat) — used for the guardrail thresholds.
# ---------------------------------------------------------------------------

BASE_1182 = dict(prec_x=2.05, r_dot_x=0.901, r_trk_x=0.550, valid=0.70)   # physics N=300
SOTA_1182 = dict(prec_x=1.98, r_dot_x=0.905, r_trk_x=0.551, valid=0.71)   # learned N=1000
BASE_11823 = dict(prec_x=1.66, r_dot_x=0.797, r_trk_x=0.553, valid=0.60)  # physics N=300 dur20
SOTA_11823 = dict(prec_x=1.62, r_dot_x=0.797, r_trk_x=0.553, valid=0.60)  # learned N=1000 dur20

R_DOT_TOL = 0.005
R_TRK_TOL = 0.005
VALID_TOL = 0.02


def _eval(r, label, t_lo=None, t_hi=None):
    """Evaluate a reconstruction dict, optionally restricted to a time window
    [t_lo, t_hi) for the held-out split. Returns the khz2d eval dict + a compact
    summary subset."""
    t = np.asarray(r["t"], float)
    valid = r["valid"].astype(bool)
    if t_lo is not None or t_hi is not None:
        m = np.ones(len(t), bool)
        if t_lo is not None:
            m &= t >= t_lo
        if t_hi is not None:
            m &= t < t_hi
        valid = valid & m
    ev = khz2d.evaluate(t, r["x_px"], r["y_px"], valid, float(r["rate"]),
                        label, smooth_ms=2)
    return ev


def _row(name, ev):
    return dict(name=name, rate=ev["rate"], prec_x=ev["prec_x"],
                r_dot_x=ev["r_dot_x"], r_trk_x=ev["r_trk_x"],
                prec_y=ev["prec_y"], valid=ev["valid_frac"])


def _passes_guardrail(row, base):
    return (row["r_dot_x"] >= base["r_dot_x"] - R_DOT_TOL and
            row["r_trk_x"] >= base["r_trk_x"] - R_TRK_TOL and
            row["valid"] >= base["valid"] - VALID_TOL)


# ---------------------------------------------------------------------------
# Config grid. Each config is a dict of m4_dpf kwargs (minus eff_rate/dur_s).
# One-at-a-time variations from the physics N=300 baseline + a learned arm.
# ---------------------------------------------------------------------------


def broad_configs():
    base = dict(likelihood="physics", n_particles=300)
    cfgs = []

    def add(name, **kw):
        c = dict(base); c.update(kw); cfgs.append((name, c))

    # --- baseline + the existing learned SOTA ---
    add("phys_N300_base")
    add("learn_N1000", likelihood="learned", n_particles=1000)

    # --- particle count (physics) ---
    add("phys_N500", n_particles=500)
    add("phys_N1000", n_particles=1000)
    add("phys_N2000", n_particles=2000)

    # --- BETA (physics only; inert under learned) ---
    add("phys_b10", beta=10.0)
    add("phys_b30", beta=30.0)
    add("phys_b40", beta=40.0)

    # --- HP_SIGMA: the fine appearance band — flagged high-leverage on real ---
    add("phys_hp3", hp_sigma=3.0)
    add("phys_hp4", hp_sigma=4.0)
    add("phys_hp5", hp_sigma=5.0)
    add("phys_hp8", hp_sigma=8.0)
    add("phys_hp10", hp_sigma=10.0)
    add("phys_hp12", hp_sigma=12.0)

    # --- SIGMA_ALONG (trusted-along tightness) ---
    add("phys_sa1", sigma_along=1.0)
    add("phys_sa3", sigma_along=3.0)

    # --- ESS_FRAC ---
    add("phys_ess0.3", ess_frac=0.3)
    add("phys_ess0.7", ess_frac=0.7)

    # --- roughening ---
    add("phys_rp0.25", roughen_perp=0.25)
    add("phys_rp1.0", roughen_perp=1.0)

    # --- reacquisition window ---
    add("phys_nw3", ncc_loss_window=3)
    add("phys_nw10", ncc_loss_window=10)

    # --- reseed spread (m4 default 30; try alias-spacing & intermediate) ---
    add("phys_rs125", reseed_perp_sigma=125.0)
    add("phys_rs60", reseed_perp_sigma=60.0)

    # --- geometry ---
    add("phys_ll250", line_len=250)
    add("phys_ll300", line_len=300)
    add("phys_pw120", padw=120)

    return cfgs


def combine_configs():
    """Phase-2: push the HP trend higher and combine the precision-helping
    single-lever winners (nw3, ess0.7, hp-high, rp0.25), for both physics and
    the learned head (nw/ess affect both; hp_sigma affects the physics weight
    and — under the learned head — only the lock monitor)."""
    cfgs = []

    def add(name, **kw):
        cfgs.append((name, kw))

    # extend the monotonic HP trend
    add("phys_hp16", likelihood="physics", n_particles=300, hp_sigma=16.0)
    add("phys_hp20", likelihood="physics", n_particles=300, hp_sigma=20.0)
    # physics combos of the precision-helping levers
    add("phys_nw3_ess0.7", likelihood="physics", n_particles=300,
        ncc_loss_window=3, ess_frac=0.7)
    add("phys_nw3_hp16", likelihood="physics", n_particles=300,
        ncc_loss_window=3, hp_sigma=16.0)
    add("phys_nw3_ess0.7_hp16", likelihood="physics", n_particles=300,
        ncc_loss_window=3, ess_frac=0.7, hp_sigma=16.0)
    add("phys_nw3_ess0.7_hp16_rp0.25", likelihood="physics", n_particles=300,
        ncc_loss_window=3, ess_frac=0.7, hp_sigma=16.0, roughen_perp=0.25)
    # learned head + the winning stability levers
    add("learn_nw3", likelihood="learned", n_particles=1000, ncc_loss_window=3)
    add("learn_nw3_ess0.7", likelihood="learned", n_particles=1000,
        ncc_loss_window=3, ess_frac=0.7)
    return cfgs


def run_grid(cfgs, eff_rate, dur_s=None, rebuild=False, base=BASE_1182,
             held_split=None):
    """Run every config, evaluate full + (optional) held-out halves, return rows."""
    rows = []
    for i, (name, kw) in enumerate(cfgs):
        t0 = time.time()
        print(f"\n=== [{i+1}/{len(cfgs)}] {name}  (rate={eff_rate}, dur={dur_s}) ===")
        try:
            r = M.m4_dpf(eff_rate, dur_s=dur_s, rebuild=rebuild,
                         tag_extra="", **kw)
        except Exception as e:
            print(f"  !! {name} ERRORED: {e}")
            continue
        ev = _eval(r, name)
        row = _row(name, ev)
        row["sec"] = time.time() - t0
        row["kw"] = kw
        # held-out split: pick window vs validate window
        if held_split is not None:
            tA, tB = held_split
            evA = _eval(r, name + "_A", t_lo=None, t_hi=tA)
            evB = _eval(r, name + "_B", t_lo=tB, t_hi=None)
            row["prec_x_A"] = evA["prec_x"]
            row["prec_x_B"] = evB["prec_x"]
            row["r_trk_x_B"] = evB["r_trk_x"]
            row["r_dot_x_B"] = evB["r_dot_x"]
        row["pass"] = _passes_guardrail(row, base)
        rows.append(row)
        print(f"  prec_x={row['prec_x']:.3f} r_dot_x={row['r_dot_x']:.3f} "
              f"r_trk_x={row['r_trk_x']:.3f} valid={row['valid']*100:.0f}% "
              f"{'PASS' if row['pass'] else 'fail-guardrail'} ({row['sec']:.0f}s)")
    return rows


def print_leaderboard(rows, base, title):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    # sort: guardrail-passing first, then by prec_x ascending
    rows_sorted = sorted(rows, key=lambda r: (not r["pass"], r["prec_x"]))
    hdr = (f"{'config':<22s} {'prec_x':>7s} {'r_dot_x':>8s} {'r_trk_x':>8s} "
           f"{'prec_y':>7s} {'valid':>6s} {'guard':>6s}")
    print(hdr); print("-" * len(hdr))
    for r in rows_sorted:
        flag = "PASS" if r["pass"] else "fail"
        win = "*" if (r["pass"] and r["prec_x"] < base["prec_x"]) else " "
        print(f"{r['name']:<22s} {r['prec_x']:>7.3f} {r['r_dot_x']:>8.3f} "
              f"{r['r_trk_x']:>8.3f} {r['prec_y']:>7.3f} {r['valid']*100:>5.0f}% "
              f"{flag:>6s}{win}")
    return rows_sorted


def md_table(rows_sorted, base):
    lines = ["| config | prec_x (') | r_dot_x | r_trk_x | prec_y (') | valid | guardrail |",
             "|---|---|---|---|---|---|---|"]
    for r in rows_sorted:
        g = "PASS" if r["pass"] else "fail"
        if r["pass"] and r["prec_x"] < base["prec_x"]:
            g = "**PASS+win**"
        lines.append(f"| {r['name']} | {r['prec_x']:.3f} | {r['r_dot_x']:.3f} | "
                     f"{r['r_trk_x']:.3f} | {r['prec_y']:.3f} | "
                     f"{r['valid']*100:.0f}% | {g} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report: results/real_eye_optimization.md + .png from the saved rows alone.
# ---------------------------------------------------------------------------


def make_figure(rows_1182, rows_11823, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e3da"
    ACCENT, BLUE, GOLD = "#9c1f2e", "#22507a", "#c08a2e"
    matplotlib.rcParams.update({
        "figure.dpi": 220, "savefig.dpi": 220, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.edgecolor": INK, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.labelsize": 7.2,
        "ytick.labelsize": 8, "xtick.color": INK, "ytick.color": INK,
        "legend.fontsize": 8, "legend.frameon": False, "text.color": INK,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4),
                             width_ratios=[1.55, 1.0, 1.0])

    def color_of(name):
        return BLUE if name.startswith("learn") else ACCENT

    # --- panel 1: 1182 Hz leaderboard (top 12 + baseline) ---
    ax = axes[0]
    top = [r for r in rows_1182 if r["pass"]][:11]
    if not any(r["name"] == "phys_N300_base" for r in top):
        top += [r for r in rows_1182 if r["name"] == "phys_N300_base"]
    names = [r["name"] for r in top]
    vals = [r["prec_x"] for r in top]
    cols = [color_of(n) for n in names]
    ax.bar(range(len(top)), vals, color=cols, width=0.7)
    ax.axhline(BASE_1182["prec_x"], color=MUTED, lw=0.9, ls=(0, (3, 2)))
    ax.text(len(top) - 0.4, BASE_1182["prec_x"] + 0.012,
            "previous physics N=300", fontsize=6.6, color=MUTED, ha="right")
    ax.axhline(SOTA_1182["prec_x"], color=GOLD, lw=0.9, ls=(0, (3, 2)))
    ax.text(len(top) - 0.4, SOTA_1182["prec_x"] - 0.045,
            "SOTA learned N=1000", fontsize=6.6, color=GOLD, ha="right")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylabel("precision x (arcmin)")
    ax.set_ylim(min(vals) - 0.12, max(vals) + 0.08)
    ax.set_title("1182 Hz full-length — guardrail-valid leaderboard")

    # --- panel 2: held-out halves for the leaders ---
    ax = axes[1]
    show = [r for r in rows_1182
            if r["name"] in ("phys_N300_base", "learn_N1000", "phys_nw3",
                             "learn_nw3", "learn_nw3_ess0.7")]
    show = sorted(show, key=lambda r: r["prec_x"], reverse=True)
    y = np.arange(len(show))
    ax.barh(y + 0.18, [r["prec_x_A"] for r in show], height=0.34,
            color=BLUE, label="pick half (t<35 s)")
    ax.barh(y - 0.18, [r["prec_x_B"] for r in show], height=0.34,
            color=ACCENT, label="held-out half (t\u226535 s)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in show], fontsize=7.2)
    ax.set_xlabel("precision x (arcmin)")
    ax.set_title("held-out check @1182 Hz")
    ax.legend(loc="lower right", fontsize=6.6)

    # --- panel 3: line-rate validation ---
    ax = axes[2]
    names = [r["name"] for r in rows_11823]
    vals = [r["prec_x"] for r in rows_11823]
    cols = [color_of(n) for n in names]
    order = np.argsort(vals)[::-1]
    ax.bar(range(len(names)), [vals[i] for i in order],
           color=[cols[i] for i in order], width=0.65)
    ax.axhline(BASE_11823["prec_x"], color=MUTED, lw=0.9, ls=(0, (3, 2)))
    ax.axhline(SOTA_11823["prec_x"], color=GOLD, lw=0.9, ls=(0, (3, 2)))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[i] for i in order], rotation=38, ha="right")
    ax.set_ylabel("precision x (arcmin)")
    ax.set_ylim(min(vals) - 0.08, max(vals) + 0.05)
    ax.set_title("11823 Hz line rate (dur=20 s, matched)")

    fig.savefig(out_png)
    plt.close(fig)
    print("wrote", out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="broad",
                    choices=["broad", "combine", "validate", "report", "all"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    if a.phase in ("broad", "all"):
        cfgs = broad_configs()
        # held-out: pick configs on t<35s (A), validate on t>=35s (B)
        rows = run_grid(cfgs, 1182.0, dur_s=None, rebuild=a.rebuild,
                        base=BASE_1182, held_split=(35.0, 35.0))
        rows_sorted = print_leaderboard(rows, BASE_1182,
                                        "BROAD SEARCH — 1182 Hz full-length")
        np.save("cache/_optimize_broad_rows.npy",
                np.array(rows_sorted, dtype=object), allow_pickle=True)

    if a.phase in ("combine", "all"):
        cfgs = combine_configs()
        rows = run_grid(cfgs, 1182.0, dur_s=None, rebuild=a.rebuild,
                        base=BASE_1182, held_split=(35.0, 35.0))
        # fold in the broad rows for a unified leaderboard
        try:
            prev = list(np.load("cache/_optimize_broad_rows.npy", allow_pickle=True))
        except Exception:
            prev = []
        allrows = prev + rows
        rows_sorted = print_leaderboard(allrows, BASE_1182,
                                        "COMBINED LEADERBOARD — 1182 Hz full-length")
        np.save("cache/_optimize_broad_rows.npy",
                np.array(rows_sorted, dtype=object), allow_pickle=True)

    if a.phase in ("validate", "all"):
        # Validate the top guardrail-passing configs at the 11823 Hz line rate
        # over a dur=20 s window (matched to the published SOTA comparison).
        rows_sorted = list(np.load("cache/_optimize_broad_rows.npy",
                                   allow_pickle=True))
        # always include the two reference arms
        ref = [("phys_N300_base", dict(likelihood="physics", n_particles=300)),
               ("learn_N1000", dict(likelihood="learned", n_particles=1000))]
        # top physics + combo winners that PASS guardrail, best prec first
        winners = [(r["name"], r["kw"]) for r in rows_sorted
                   if r["pass"] and r["name"] not in ("phys_N300_base", "learn_N1000")]
        picks, seen = [], set()
        for name, kw in ref + winners[:4]:
            key = name
            if key in seen:
                continue
            seen.add(key)
            picks.append((name, kw))
        print(f"\nVALIDATE picks @11823 dur=20: {[p[0] for p in picks]}")
        vrows = run_grid(picks, 11823.0, dur_s=20.0, rebuild=a.rebuild,
                         base=BASE_11823)
        print_leaderboard(vrows, BASE_11823,
                          "VALIDATION — 11823 Hz (dur=20 s, matched)")
        np.save("cache/_optimize_validate_rows.npy",
                np.array(vrows, dtype=object), allow_pickle=True)

    if a.phase == "report":
        # Cache-only: rebuild the report from the saved row files.
        rows_1182 = list(np.load("cache/_optimize_broad_rows.npy",
                                 allow_pickle=True))
        rows_1182 = sorted(rows_1182, key=lambda r: (not r["pass"], r["prec_x"]))
        rows_11823 = list(np.load("cache/_optimize_validate_rows.npy",
                                  allow_pickle=True))
        print_leaderboard(rows_1182, BASE_1182, "1182 Hz full-length")
        print_leaderboard(rows_11823, BASE_11823, "11823 Hz dur=20 s")
        make_figure(rows_1182, rows_11823, "results/real_eye_optimization.png")
