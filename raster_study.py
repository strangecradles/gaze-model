"""raster_study.py — best-possible 2D gaze from the test1 SLO raster.

Workhorse = the incremental strip tracker (raster_track.track, ref_mode
'incremental'): it recovers 2D gaze at strip rate = (808 / S) * 14.633 Hz, far
above the 14.6 Hz frame rate. This module measures, on REAL test1:

  1. per-stimulus-phase tracking fidelity (Pearson r) vs BOTH the pursuit dot
     (the target) AND the independent ~32.5 Hz machine tracker (a separate
     measurement path) — the honest dual reference, broken out per driven axis;
  2. the absolute px->arcmin scale by per-phase regression vs the tracker
     (along from H_sine, perp from V_sine);
  3. a TRUTH-FREE precision floor: RMS of the gaze trace high-passed above the
     oculomotor band (>~40 Hz), where the eye contributes ~no power, so it is
     dominated by per-strip registration noise (the Sheehy "residual motion
     after stabilization" analog);
  4. the rate->accuracy frontier as S sweeps 32..1 (732 Hz .. 11.8 kHz line rate)

and writes results/raster_rate_accuracy.{png,md}. Absolute labeled accuracy is
certified separately on synthetic streams (raster_synth.py) — real test1 has no
high-rate ground truth, so correlation + a truth-free noise floor + independent-
tracker agreement is the honest real-data triangulation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter

import data
import raster_track as rt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

PHASE_AXIS = {"H_sine": "x", "V_sine": "y", "circle": "xy", "lissajous": "xy"}


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else np.nan


def _fill(x):
    x = np.asarray(x, float); i = np.arange(len(x)); m = np.isfinite(x)
    return np.interp(i, i[m], x[m]) if m.any() else x


@dataclass
class Refs:
    t_stim: np.ndarray
    dotx: np.ndarray      # arcmin
    doty: np.ndarray
    phase: np.ndarray     # str per stim sample
    t_trk: np.ndarray
    trkx: np.ndarray      # arcmin
    trky: np.ndarray


def load_refs(which="test1") -> Refs:
    st = data.load_stimulus("pursuit")
    ph = pd.read_csv(os.path.join(HERE, "stimuli", "pursuit.csv"))["phase"].to_numpy().astype(str)
    tr = data.load_tracker(which)
    return Refs(st.t_s, st.x_deg * 60.0, st.y_deg * 60.0, ph,
               tr.t_s - tr.t_s[0],
               _fill(tr.right_x) * data.UNITS.tracker_deg_per_unit_x * 60.0,
               _fill(tr.right_y) * data.UNITS.tracker_deg_per_unit_y * 60.0)


def _detrend(x, strip_hz, f_lo=0.05):
    """Remove drift below f_lo Hz (keep the 0.2 Hz pursuit band + everything above)."""
    x = np.asarray(x, float)
    return x - gaussian_filter1d(x, strip_hz / (2 * np.pi * f_lo))


def joint_offset(t, H, V, fov, refs, lo=0.5, hi=4.0, n=141):
    best = (-9.0, 2.2)
    for o in np.linspace(lo, hi, n):
        dx = np.interp(t, refs.t_stim + o, refs.dotx)
        dy = np.interp(t, refs.t_stim + o, refs.doty)
        s = abs(np.nan_to_num(_corr(H[fov], dx[fov]))) + abs(np.nan_to_num(_corr(V[fov], dy[fov])))
        if s > best[0]:
            best = (s, float(o))
    return best[1]


def _phase_mask(t, off, refs, name):
    lab = (refs.phase == name).astype(float)
    return np.interp(t, refs.t_stim + off, lab) > 0.5


@dataclass
class Scale:
    along: float = np.nan   # arcmin/px (horizontal/slow axis)
    perp: float = np.nan    # arcmin/px (vertical/fast axis)
    r_along: float = np.nan
    r_perp: float = np.nan


def measure_scale(t, H, V, fov, off, refs) -> Scale:
    """Per-phase robust scale vs the independent tracker (sign from r, std ratio).
    along from H_sine (pure horizontal), perp from V_sine (pure vertical)."""
    mx = np.interp(t, refs.t_trk + off, refs.trkx)
    my = np.interp(t, refs.t_trk + off, refs.trky)
    h = _phase_mask(t, off, refs, "H_sine") & fov
    v = _phase_mask(t, off, refs, "V_sine") & fov
    sc = Scale()
    if h.sum() > 50:
        sc.r_along = _corr(H[h], mx[h])
        sc.along = float(np.nanstd(mx[h]) / (np.nanstd(H[h]) + 1e-9))
    if v.sum() > 50:
        sc.r_perp = _corr(V[v], my[v])
        sc.perp = float(np.nanstd(my[v]) / (np.nanstd(V[v]) + 1e-9))
    return sc


def noise_floor_arcmin(t, sig_px, fov, strip_hz, scale_arcmin_per_px, f_hi=40.0):
    """RMS of the signal high-passed above f_hi Hz (eye has ~no power there),
    in arcmin. Truth-free per-sample registration-noise floor."""
    x = np.asarray(sig_px, float).copy()
    x[~fov] = np.nan
    x = _fill(x)
    hp = x - gaussian_filter1d(x, strip_hz / (2 * np.pi * f_hi))
    return float(np.nanstd(hp[fov]) * scale_arcmin_per_px)


@dataclass
class RateRow:
    S: int
    rate: float
    infov: float
    r_dot: dict = field(default_factory=dict)   # 'H_sine_x', 'V_sine_y', ...
    r_trk: dict = field(default_factory=dict)
    nf_perp: float = np.nan      # arcmin
    nf_along: float = np.nan


def analyze(S, which="test1", max_frames=None, pad=rt.PAD, scale: Scale = None) -> tuple:
    tk = rt.track(which, S=S, ref_mode="incremental", max_frames=max_frames, pad=pad)
    refs = load_refs(which)
    fov = tk.infov
    H = _detrend(median_filter(tk.along_px, 5), tk.strip_hz)
    V = _detrend(median_filter(tk.perp_px, 5), tk.strip_hz)
    off = joint_offset(tk.t, H, V, fov, refs)
    if scale is None:
        scale = measure_scale(tk.t, H, V, fov, off, refs)
    dx = np.interp(tk.t, refs.t_stim + off, refs.dotx)
    dy = np.interp(tk.t, refs.t_stim + off, refs.doty)
    mx = np.interp(tk.t, refs.t_trk + off, refs.trkx)
    my = np.interp(tk.t, refs.t_trk + off, refs.trky)
    row = RateRow(S, tk.strip_hz, float(fov.mean()))
    for name, axis in PHASE_AXIS.items():
        m = _phase_mask(tk.t, off, refs, name) & fov
        if "x" in axis:
            row.r_dot[f"{name}_x"] = _corr(H[m], dx[m]); row.r_trk[f"{name}_x"] = _corr(H[m], mx[m])
        if "y" in axis:
            row.r_dot[f"{name}_y"] = _corr(V[m], dy[m]); row.r_trk[f"{name}_y"] = _corr(V[m], my[m])
    sa = scale.along if np.isfinite(scale.along) else 0.07
    sp = scale.perp if np.isfinite(scale.perp) else 0.07
    row.nf_along = noise_floor_arcmin(tk.t, tk.along_px, fov, tk.strip_hz, sa)
    row.nf_perp = noise_floor_arcmin(tk.t, tk.perp_px, fov, tk.strip_hz, sp)
    return row, scale, off


def sweep(S_list=(32, 16, 8, 4, 2), which="test1", max_frames=None, pad=rt.PAD):
    # scale measured once on a clean mid-rate run, reused (px->arcmin is physical)
    _, scale, _ = analyze(8, which, max_frames, pad)
    rows = []
    for S in S_list:
        row, _, off = analyze(S, which, max_frames, pad, scale=scale)
        rows.append(row)
        print(f"S={S:>2} {row.rate:6.0f}Hz inFOV={row.infov*100:3.0f}%  "
              f"Hx={row.r_dot.get('H_sine_x', float('nan')):+.2f} "
              f"Vy={row.r_dot.get('V_sine_y', float('nan')):+.2f} "
              f"Cx={row.r_dot.get('circle_x', float('nan')):+.2f} "
              f"Cy={row.r_dot.get('circle_y', float('nan')):+.2f}  "
              f"NF perp={row.nf_perp:.2f}' along={row.nf_along:.2f}'", flush=True)
    return rows, scale


def figure(rows, scale, path=os.path.join(RESULTS, "raster_rate_accuracy.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rates = [r.rate for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    # panel 1: per-phase r_dot vs rate
    for key, lab, c in [("H_sine_x", "H_sine horiz", "C0"), ("V_sine_y", "V_sine vert", "C3"),
                        ("circle_x", "circle horiz", "C0"), ("circle_y", "circle vert", "C3"),
                        ("lissajous_x", "liss horiz", "C0"), ("lissajous_y", "liss vert", "C3")]:
        ls = "-" if "circle" in key else ("--" if "liss" in key else ":")
        y = [r.r_dot.get(key, np.nan) for r in rows]
        ax[0].plot(rates, y, ls, color=c, marker="o", ms=4, alpha=.8, label=lab)
    ax[0].axvline(1000, color="k", lw=1, ls=":"); ax[0].text(1050, 0.1, "1 kHz", rotation=90, va="bottom")
    ax[0].set_xscale("log"); ax[0].set_xlabel("strip rate (Hz)"); ax[0].set_ylabel("r vs pursuit dot")
    ax[0].set_title("Tracking fidelity vs rate (per phase/axis)"); ax[0].set_ylim(0, 1)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=7, ncol=2)
    # panel 2: noise floor arcmin vs rate
    ax[1].plot(rates, [r.nf_perp for r in rows], "o-", color="C3", label="vertical (fast)")
    ax[1].plot(rates, [r.nf_along for r in rows], "o-", color="C0", label="horizontal (slow)")
    ax[1].axvline(1000, color="k", lw=1, ls=":"); ax[1].axhline(1.0, color="0.5", lw=1, ls="--")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("strip rate (Hz)"); ax[1].set_ylabel("noise floor (' arcmin, >40 Hz RMS)")
    ax[1].set_title("Truth-free precision floor vs rate"); ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
    # panel 3: in-FOV / lock vs rate
    ax[2].plot(rates, [r.infov * 100 for r in rows], "o-", color="C2")
    ax[2].axvline(1000, color="k", lw=1, ls=":")
    ax[2].set_xscale("log"); ax[2].set_xlabel("strip rate (Hz)"); ax[2].set_ylabel("in-FOV / locked (%)")
    ax[2].set_title("Lock rate vs rate"); ax[2].grid(alpha=.3); ax[2].set_ylim(0, 100)
    fig.suptitle(f"test1 SLO raster — 2D gaze rate/accuracy frontier (incremental strip tracking)  "
                 f"|  scale ~{scale.along:.3f}'/px along, {scale.perp:.3f}'/px perp", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_md(rows, scale, path=os.path.join(RESULTS, "raster_rate_accuracy.md")):
    L = ["# test1 SLO raster — 2D gaze rate/accuracy frontier", ""]
    L.append(f"Method: incremental strip registration (TSLO-style). "
             f"Scale (per-phase vs 32.5 Hz tracker): along {scale.along:.3f}'/px (r={scale.r_along:+.2f}), "
             f"perp {scale.perp:.3f}'/px (r={scale.r_perp:+.2f}).")
    L.append("")
    L.append("| S | rate (Hz) | in-FOV % | r_dot Hx | r_dot Vy | r_dot Cx | r_dot Cy | r_dot Lx | r_dot Ly | NF perp (') | NF along (') |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        g = lambda k: r.r_dot.get(k, float("nan"))
        L.append(f"| {r.S} | {r.rate:.0f} | {r.infov*100:.0f} | {g('H_sine_x'):+.2f} | {g('V_sine_y'):+.2f} | "
                 f"{g('circle_x'):+.2f} | {g('circle_y'):+.2f} | {g('lissajous_x'):+.2f} | {g('lissajous_y'):+.2f} | "
                 f"{r.nf_perp:.2f} | {r.nf_along:.2f} |")
    L.append("")
    L.append("r_dot = Pearson vs pursuit dot (target); H/V/C/L = H_sine/V_sine/circle/lissajous; "
             "x = horizontal (slow axis), y = vertical (fast axis). NF = truth-free >40 Hz noise floor.")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, nargs="+", default=[32, 16, 8, 4, 2])
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--pad", type=int, default=rt.PAD)
    a = p.parse_args()
    rows, scale = sweep(tuple(a.S), max_frames=a.max_frames, pad=a.pad)
    fp = figure(rows, scale); mp = write_md(rows, scale)
    print(f"wrote {fp}\nwrote {mp}")


if __name__ == "__main__":
    main()
