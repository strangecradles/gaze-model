"""gaze_anim_methods.py — one gaze-vs-dot animation per kHz-2D-study method,
in the results/gaze2d_anim.mp4 style (see gaze_anim_m4.py for the layout).

Renders every cached testbed-A method (results/khz2d_methods.md) to
results/animations/gaze2d_<method>.mp4. Layout: left = 2D plane (dot star +
machine tracker + recon dot, fading tails); right = horizontal/vertical time
series with a moving cursor and out-of-FOV shading; dot-referenced arcmin
frame; real-time playback. Display calibration = sign + amplitude match to the
dot (gaze_anim.py convention); honest r values on the panels.

Usage:
  python gaze_anim_methods.py --method m3_viterbi
  python gaze_anim_methods.py --list
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import median_filter

import khz2d

OUTDIR = os.path.join(khz2d.RESULTS, "animations")
VIDEO_FPS = 30.0
TAIL_S = 0.8

LABELS = {
    "m0_chain":            "M0 — 2D SLO frames only (15 Hz baseline)",
    "m1_s20":              "M1 — strip tracking, S=20 cols",
    "m1_s8":               "M1 — strip tracking, S=8 cols",
    "m1_s4":               "M1 — strip tracking, S=4 cols",
    "m1_s2":               "M1 — strip tracking, S=2 cols",
    "m1_s1":               "M1 — strip tracking, S=1 col (line rate)",
    "m2_kalman":           "M2 — per-line Kalman fusion",
    "m3_viterbi":          "M3 — Viterbi decode between frames",
    "m4_dpf_1182":         "M4 — particle filter @ 1.2 kHz",
    "m4_dpf_11823":        "M4 — particle filter @ line rate (strongest)",
    "m4_dpf_1182_learned": "M4 — particle filter, learned likelihood",
    "m5_map":              "M5 — batch MAP smoother",
}


def render(method: str, out: str | None = None):
    r = khz2d.load_method(method)
    if r is None:
        raise SystemExit(f"{method} not cached — run khz2d_methods.py first")
    out = out or os.path.join(OUTDIR, f"gaze2d_{method}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    label = LABELS.get(method, method)
    rate = float(r["rate"])
    t = np.asarray(r["t"], float)
    valid = r["valid"].astype(bool)
    T = len(t)

    ev = khz2d.evaluate(t, r["x_px"], r["y_px"], valid, rate, method, smooth_ms=2)
    R = khz2d.refs()
    DX, DY = ev["dot_x"], ev["dot_y"]
    MX = np.interp(t, R["trk_t"] + R["off"], median_filter(R["trk_x"], 5))
    MY = np.interp(t, R["trk_t"] + R["off"], median_filter(R["trk_y"], 5))
    MX = MX - MX.mean() + np.nanmean(DX)
    MY = MY - MY.mean() + np.nanmean(DY)

    def disp(cal, ref):
        v = valid & np.isfinite(cal)
        rr = khz2d.corr(cal[v], ref[v])
        s = (1.0 if rr >= 0 else -1.0) * (ref[v].std() / (cal[v].std() + 1e-9))
        return s * (cal - cal[v].mean()) + ref[v].mean()
    RX = disp(ev["cal_x"], DX); RY = disp(ev["cal_y"], DY)
    RXm = np.where(valid, RX, np.nan); RYm = np.where(valid, RY, np.nan)
    print(f"[anim:{method}] {rate:.0f} Hz, {T} samples ({t[-1]-t[0]:.1f}s), "
          f"in-FOV {valid.mean()*100:.0f}%, r x={ev['r_dot_x']:.2f} y={ev['r_dot_y']:.2f}")

    cx, cy = np.nanmean(DX), np.nanmean(DY)
    DXc, DYc, MXc, MYc = DX - cx, DY - cy, MX - cx, MY - cy
    RXc, RYc = RXm - cx, RYm - cy

    step = max(1, int(round(rate / VIDEO_FPS)))
    fps = rate / step
    tail = max(2, int(TAIL_S * rate))
    fr = np.arange(0, T, step)
    # rate-adaptive decimation for static/tail line plots
    bg_d = max(1, int(rate / 500))
    ts_d = max(1, int(rate / 1500))
    tl_d = max(1, int(rate / 3000))

    fig = plt.figure(figsize=(14, 7))
    axp = fig.add_subplot(1, 2, 1)
    axp.plot(DXc[::bg_d], DYc[::bg_d], color='C2', lw=.3, alpha=.25)
    axp.plot(RXc[::bg_d], RYc[::bg_d], color='C3', lw=.3, alpha=.18)
    lim = max(np.nanpercentile(np.abs(np.r_[DXc, DYc]), 99), 20) * 1.2
    axp.set_xlim(-lim, lim); axp.set_ylim(-lim, lim)
    axp.set_aspect('equal'); axp.grid(alpha=.3)
    axp.set_xlabel("horizontal (arcmin)"); axp.set_ylabel("vertical (arcmin)")
    axp.set_title(f"2D gaze @ {rate:.0f} Hz (in-FOV only)")
    dtail, = axp.plot([], [], color='C2', lw=1.4, alpha=.5)
    mtail, = axp.plot([], [], color='C0', lw=1.2, alpha=.4)
    rtail, = axp.plot([], [], color='C3', lw=1.2, alpha=.6)
    ddot, = axp.plot([], [], '*', color='C2', ms=20, label='dot (target)')
    mdot, = axp.plot([], [], 'o', color='C0', ms=10, label='machine tracker (32.5 Hz)')
    rdot, = axp.plot([], [], 'o', color='C3', ms=9, label=f'recon ({rate:.0f} Hz)')
    axp.legend(loc='upper right', fontsize=9)
    ttxt = axp.text(.03, .97, "", transform=axp.transAxes, va='top',
                    fontsize=11, family='monospace')

    def ts_panel(pos, D, M, Rm, ylab, title):
        ax = fig.add_subplot(2, 2, pos)
        ax.plot(t[::ts_d], D[::ts_d], color='C2', lw=1.0, alpha=.7, label='dot')
        ax.plot(t[::ts_d], M[::ts_d], color='C0', lw=.5, alpha=.6, label='machine')
        ax.plot(t[::tl_d], Rm[::tl_d], color='C3', lw=.4, label='recon')
        ylo = np.nanpercentile(np.r_[D, Rm[valid]], 0.5) - 15
        yhi = np.nanpercentile(np.r_[D, Rm[valid]], 99.5) + 15
        ax.fill_between(t[::ts_d], ylo, yhi, where=~valid[::ts_d],
                        color='0.85', alpha=.5, lw=0, step='mid')
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=10); ax.grid(alpha=.3)
        return ax, ax.axvline(t[0], color='k', lw=1)
    axx, cvx = ts_panel(2, DXc, MXc, RXc, "horiz (')",
                        f"Horizontal  recon-vs-dot r={ev['r_dot_x']:+.2f} "
                        f"(vs tracker {ev['r_trk_x']:+.2f})")
    axx.legend(fontsize=7, loc='upper right')
    axy, cvy = ts_panel(4, DYc, MYc, RYc, "vert (')",
                        f"Vertical  recon-vs-dot r={ev['r_dot_y']:+.2f} "
                        f"(vs tracker {ev['r_trk_y']:+.2f})")
    axy.set_xlabel("time (s)")
    fig.suptitle(f"kHz 2D gaze from x-scan lines + 2D SLO — {label}  |  "
                 f"{rate:.0f} Hz  |  in-FOV {valid.mean()*100:.0f}%  "
                 f"(results/khz2d_methods.md)", fontsize=11, y=.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def upd(i):
        j = fr[i]; a = max(0, j - tail)
        dtail.set_data(DXc[a:j+1:tl_d], DYc[a:j+1:tl_d])
        mtail.set_data(MXc[a:j+1:tl_d], MYc[a:j+1:tl_d])
        rtail.set_data(RXc[a:j+1:tl_d], RYc[a:j+1:tl_d])
        ddot.set_data([DXc[j]], [DYc[j]]); mdot.set_data([MXc[j]], [MYc[j]])
        if valid[j]:
            rdot.set_data([RXc[j]], [RYc[j]])
            s = f"err={np.hypot(RXc[j]-DXc[j], RYc[j]-DYc[j]):4.0f}'"
        else:
            rdot.set_data([], []); s = "OUT OF FOV"
        ttxt.set_text(f"t={t[j]:6.2f}s\n{s}")
        cvx.set_xdata([t[j], t[j]]); cvy.set_xdata([t[j], t[j]])
        return dtail, mtail, rtail, ddot, mdot, rdot, ttxt, cvx, cvy

    ani = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000 / fps,
                                  blit=True)
    ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    print(f"[anim:{method}] wrote {out} ({len(fr)} frames @ {fps:.1f} fps)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or a.method is None:
        for m in LABELS:
            print(("cached  " if khz2d.load_method(m) is not None else "MISSING ") + m)
    else:
        render(a.method)
