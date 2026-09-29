"""people_data_fov_anim.py — render event-preserving gaze animations from cache.

Reads cached M4 particle-filter outputs (cache/people_fov/<name>/) and writes
animations to people_data_results/. Raw line-rate traces stay in cache; the
display path uses oculomotor_trajectory() so short mosaic-alias excursions are
repaired, pursuit looks smooth, and saccades (plus tremor above the noise floor)
remain visible.

Usage::
    python people_data_fov_anim.py                    # all cached subjects
    python people_data_fov_anim.py --person Kathy
    python people_data_fov_anim.py --style raw        # original jittery display
    python people_data_fov_anim.py --style oculomotor # default
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d, median_filter

import data
import khz2d
import oculo_smooth as osm

HERE = os.path.dirname(os.path.abspath(__file__))
PEOPLE_DIR = os.path.join(HERE, "people_data_fov")
OUT_DIR = os.path.join(HERE, "people_data_results")
CACHE_ROOT = os.path.join(HERE, "cache", "people_fov")
STIMULUS = "pursuit_fov"
ARC_PER_PX = 60.0 / 126.0
TRK_DEG_X = data.UNITS.tracker_deg_per_unit_x
TRK_DEG_Y = data.UNITS.tracker_deg_per_unit_y
VIDEO_FPS = 30.0
TAIL_S = 0.8


def discover_cached(cache_tag: str = "m4_dpf_physics") -> list[str]:
    return sorted(
        d for d in os.listdir(CACHE_ROOT)
        if os.path.isfile(os.path.join(CACHE_ROOT, d, f"{cache_tag}.npz"))
    )


def load_tracker(name: str) -> data.MachineTrack:
    csvs = glob.glob(os.path.join(PEOPLE_DIR, f"video_playback_{name}_*.csv"))
    csvs = [p for p in csvs if "SLO" not in p and "Cam" not in p]
    df = pd.read_csv(csvs[0])
    t = pd.to_datetime(df["timestamp"])
    t_s = (t - t.iloc[0]).dt.total_seconds().to_numpy().astype(np.float64)
    dur = t_s[-1] - t_s[0] if len(t_s) > 1 else 1.0
    rate = len(df) / dur if dur > 0 else float("nan")
    return data.MachineTrack(
        t_s,
        df["right_x"].astype(np.float64).to_numpy(),
        df["right_y"].astype(np.float64).to_numpy(),
        df["right_area_mm2"].astype(np.float64).to_numpy(),
        df["video_frame"].astype(np.float64).to_numpy(),
        float(rate), name)


def compute_refs(name: str, lm: dict) -> dict:
    st = data.load_stimulus(STIMULUS)
    dot_t = st.t_s.astype(float)
    dot_x = st.x_deg * 60.0
    dot_y = st.y_deg * 60.0
    tr = load_tracker(name)
    trk_t = (tr.t_s - tr.t_s[0]).astype(float)
    trk_x = khz2d.fill_nan(tr.right_x) * TRK_DEG_X * 60.0
    trk_y = khz2d.fill_nan(tr.right_y) * TRK_DEG_Y * 60.0
    pos = lm["chain_x"][lm["frame"]] + lm["rdx"]
    fs = float(lm["line_rate"])
    m = (lm["ok"].astype(bool) & (lm["qh"] > khz2d.Q_FOV)
         & (lm["con"] > khz2d.CONTRAST_FRAC * np.median(lm["con"])))
    sig = gaussian_filter1d(median_filter(khz2d.fill_nan(np.where(m, pos, np.nan)), 7),
                            fs * 0.01)
    sig = khz2d._drift_removed(np.where(m, sig, np.nan), fs)
    tt = lm["t"]
    best = (0.0, 0.0)
    for o in np.arange(-1.0, 6.0, 0.025):
        c = khz2d.corr(sig[m], np.interp(tt[m], dot_t + o, dot_x))
        if np.isfinite(c) and abs(c) > abs(best[1]):
            best = (float(o), float(c))
    return dict(dot_t=dot_t, dot_x=dot_x, dot_y=dot_y,
                trk_t=trk_t, trk_x=trk_x, trk_y=trk_y, off=best[0], off_r=best[1])


def prepare_traces(name: str, style: str, cache_tag: str = "m4_dpf_physics") -> dict:
    cache_path = os.path.join(CACHE_ROOT, name, f"{cache_tag}.npz")
    run = {k: np.load(cache_path)[k] for k in np.load(cache_path).files}
    lm = {k: np.load(os.path.join(CACHE_ROOT, name, "lines.npz"))[k]
          for k in np.load(os.path.join(CACHE_ROOT, name, "lines.npz")).files}
    refs = compute_refs(name, lm)
    t = np.asarray(run["t"], float)
    valid = run["valid"].astype(bool)
    rate = float(run["rate"])

    hx = khz2d._drift_removed(np.where(valid, run["x_px"] * ARC_PER_PX, np.nan), rate)
    hy = khz2d._drift_removed(np.where(valid, run["y_px"] * ARC_PER_PX, np.nan), rate)

    if style == "raw":
        rx, ry = hx, hy
        k_eval = max(1.0, rate * 2 / 1000)
        rx = gaussian_filter1d(khz2d.fill_nan(rx), k_eval)
        ry = gaussian_filter1d(khz2d.fill_nan(ry), k_eval)
        rx[~valid] = np.nan
        ry[~valid] = np.nan
        tag = "raw"
    else:
        rx, ry = osm.oculomotor_trajectory_2d(hx, hy, rate, valid)
        tag = "oculomotor"

    dot_x = np.interp(t, refs["dot_t"] + refs["off"], refs["dot_x"])
    dot_y = np.interp(t, refs["dot_t"] + refs["off"], refs["dot_y"])
    trk_x = np.interp(t, refs["trk_t"] + refs["off"], refs["trk_x"])
    trk_y = np.interp(t, refs["trk_t"] + refs["off"], refs["trk_y"])

    def disp(cal, ref):
        v = valid & np.isfinite(cal)
        rr = khz2d.corr(cal[v], ref[v])
        s = (1.0 if rr >= 0 else -1.0) * (ref[v].std() / (cal[v].std() + 1e-9))
        return s * (cal - cal[v].mean()) + ref[v].mean()

    rx = disp(rx, dot_x)
    ry = disp(ry, dot_y)
    rx = np.where(valid, rx, np.nan)
    ry = np.where(valid, ry, np.nan)

    return dict(name=name, tag=tag, t=t, rate=rate, valid=valid, refs=refs,
                dot_x=dot_x, dot_y=dot_y, trk_x=trk_x, trk_y=trk_y,
                rx=rx, ry=ry,
                r_dot_x=abs(khz2d.corr(rx[valid], dot_x[valid])),
                r_dot_y=abs(khz2d.corr(ry[valid], dot_y[valid])))


def render(d: dict) -> str:
    t, valid, rate = d["t"], d["valid"], d["rate"]
    refs = d["refs"]
    DX, DY = d["dot_x"], d["dot_y"]
    MX = np.interp(t, refs["trk_t"] + refs["off"], median_filter(refs["trk_x"], 5))
    MY = np.interp(t, refs["trk_t"] + refs["off"], median_filter(refs["trk_y"], 5))
    MX = MX - MX.mean() + np.nanmean(DX)
    MY = MY - MY.mean() + np.nanmean(DY)
    RXm, RYm = d["rx"], d["ry"]
    cx, cy = np.nanmean(DX), np.nanmean(DY)
    DXc, DYc = DX - cx, DY - cy
    MXc, MYc = MX - cx, MY - cy
    RXc, RYc = RXm - cx, RYm - cy

    step = max(1, int(round(rate / VIDEO_FPS)))
    fps = rate / step
    tail = int(TAIL_S * rate)
    fr = np.arange(0, len(t), step)
    bg_d = max(1, int(rate / 500))
    ts_d = max(1, int(rate / 1500))
    tl_d = max(1, int(rate / 3000))

    suffix = "_raw" if d["tag"] == "raw" else ""
    out = os.path.join(OUT_DIR, f"gaze2d_{d['name']}_m4_dpf{suffix}.mp4")
    title_style = ("raw line-rate PF" if d["tag"] == "raw"
                   else "de-aliased oculomotor (saccades + tremor kept)")

    fig = plt.figure(figsize=(14, 7))
    axp = fig.add_subplot(1, 2, 1)
    axp.plot(DXc[::bg_d], DYc[::bg_d], color="C2", lw=0.3, alpha=0.25)
    axp.plot(RXc[::bg_d], RYc[::bg_d], color="C3", lw=0.3, alpha=0.18)
    lim = max(np.nanpercentile(np.abs(np.r_[DXc, DYc]), 99), 20) * 1.2
    axp.set_xlim(-lim, lim)
    axp.set_ylim(-lim, lim)
    axp.set_aspect("equal")
    axp.grid(alpha=0.3)
    axp.set_xlabel("horizontal (arcmin)")
    axp.set_ylabel("vertical (arcmin)")
    axp.set_title(f"2D gaze — {title_style}")
    dtail, = axp.plot([], [], color="C2", lw=1.4, alpha=0.5)
    mtail, = axp.plot([], [], color="C0", lw=1.2, alpha=0.4)
    rtail, = axp.plot([], [], color="C3", lw=1.2, alpha=0.6)
    ddot, = axp.plot([], [], "*", color="C2", ms=20, label="dot (target)")
    mdot, = axp.plot([], [], "o", color="C0", ms=10, label="machine tracker")
    rdot, = axp.plot([], [], "o", color="C3", ms=9, label="recon")
    axp.legend(loc="upper right", fontsize=9)
    ttxt = axp.text(0.03, 0.97, "", transform=axp.transAxes, va="top",
                    fontsize=11, family="monospace")

    def ts_panel(pos, D, M, Rm, ylab, title):
        ax = fig.add_subplot(2, 2, pos)
        ax.plot(t[::ts_d], D[::ts_d], color="C2", lw=1.0, alpha=0.7, label="dot")
        ax.plot(t[::ts_d], M[::ts_d], color="C0", lw=0.5, alpha=0.6, label="machine")
        ax.plot(t[::tl_d], Rm[::tl_d], color="C3", lw=0.4, label="recon")
        ylo = np.nanpercentile(np.r_[D, Rm[valid]], 0.5) - 15
        yhi = np.nanpercentile(np.r_[D, Rm[valid]], 99.5) + 15
        ax.fill_between(t[::ts_d], ylo, yhi, where=~valid[::ts_d],
                        color="0.85", alpha=0.5, lw=0, step="mid")
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        return ax, ax.axvline(t[0], color="k", lw=1)

    axx, cvx = ts_panel(2, DXc, MXc, RXc, "horiz (')",
                        f"Horizontal  r={d['r_dot_x']:+.2f}")
    axx.legend(fontsize=7, loc="upper right")
    axy, cvy = ts_panel(4, DYc, MYc, RYc, "vert (')",
                        f"Vertical  r={d['r_dot_y']:+.2f}")
    axy.set_xlabel("time (s)")
    fig.suptitle(f"{d['name']} — pursuit_fov — M4 physics PF — {title_style}  |  "
                 f"in-FOV {valid.mean()*100:.0f}%", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def upd(i):
        j = fr[i]
        a = max(0, j - tail)
        dtail.set_data(DXc[a:j + 1:tl_d], DYc[a:j + 1:tl_d])
        mtail.set_data(MXc[a:j + 1:tl_d], MYc[a:j + 1:tl_d])
        rtail.set_data(RXc[a:j + 1:tl_d], RYc[a:j + 1:tl_d])
        ddot.set_data([DXc[j]], [DYc[j]])
        mdot.set_data([MXc[j]], [MYc[j]])
        if valid[j]:
            rdot.set_data([RXc[j]], [RYc[j]])
            s = f"err={np.hypot(RXc[j] - DXc[j], RYc[j] - DYc[j]):4.0f}'"
        else:
            rdot.set_data([], [])
            s = "OUT OF FOV"
        ttxt.set_text(f"t={t[j]:6.2f}s\n{s}")
        cvx.set_xdata([t[j], t[j]])
        cvy.set_xdata([t[j], t[j]])
        return dtail, mtail, rtail, ddot, mdot, rdot, ttxt, cvx, cvy

    os.makedirs(OUT_DIR, exist_ok=True)
    ani = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000 / fps, blit=True)
    ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    print(f"  [{d['name']}] wrote {out}  r x={d['r_dot_x']:.2f} y={d['r_dot_y']:.2f}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default=None)
    ap.add_argument("--style", choices=("oculomotor", "raw"), default="oculomotor")
    ap.add_argument("--cache-tag", default="m4_dpf_physics")
    args = ap.parse_args()
    names = discover_cached(args.cache_tag)
    if args.person:
        names = [n for n in names if n == args.person]
    if not names:
        raise SystemExit("no cached subjects — run the PF pipeline first")
    print(f"Rendering {len(names)} subject(s) style={args.style}: {names}")
    for name in names:
        render(prepare_traces(name, args.style, cache_tag=args.cache_tag))


if __name__ == "__main__":
    main()
