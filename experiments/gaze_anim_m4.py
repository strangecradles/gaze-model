"""gaze_anim_m4.py — gaze-vs-dot animation in the results/gaze2d_anim.mp4 style,
driven by the STRONGEST tracker from the kHz 2D study: M4, the G10-G14 particle
filter stepped per fast-axis line at the FULL 11,823 Hz line rate, fused with
the slow-axis 2D SLO frames (testbed A / test1; see results/khz2d_methods.md —
best accuracy r_dot x=0.906 y=0.752 AND best sub-frame precision 1.8').

Layout matches gaze_anim.py: left = 2D plane (dot star + machine tracker +
recon dot, fading tails); right = horizontal/vertical time-series with a moving
cursor and out-of-FOV shading; dot-referenced arcmin frame; real-time playback.

Display calibration follows gaze_anim.py's convention: sign-align + amplitude
(std) match to the dot on in-FOV samples — NOT a least-squares fit — so the
viewer sees the tracker's true relative motion. Honest r values are printed on
the panels.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import khz2d

METHOD = "m4_dpf_11823"      # the strongest tracker (full line rate DPF)
OUT = "results/gaze2d_anim_m4_khz.mp4"
VIDEO_FPS = 30.0
TAIL_S = 0.8                 # fading-tail length (s)
BG_DECIM = 24                # static background path decimation


def main():
    r = khz2d.load_method(METHOD)
    if r is None:
        raise SystemExit(f"{METHOD} not cached — run khz2d_methods.py --method m4 --rate 11823")
    rate = float(r["rate"])
    t = np.asarray(r["t"], float)
    valid = r["valid"].astype(bool)
    T = len(t)

    # shared evaluation (same protocol as the report): calibrated traces + refs
    ev = khz2d.evaluate(t, r["x_px"], r["y_px"], valid, rate, METHOD, smooth_ms=2)
    R = khz2d.refs()
    DX, DY = ev["dot_x"], ev["dot_y"]
    # tracker for display: de-spiked, re-centered into the dot frame (the eye
    # tracker has its own origin; only its relative motion is comparable)
    from scipy.ndimage import median_filter
    MX = np.interp(t, R["trk_t"] + R["off"], median_filter(R["trk_x"], 5))
    MY = np.interp(t, R["trk_t"] + R["off"], median_filter(R["trk_y"], 5))
    MX = MX - MX.mean() + np.nanmean(DX)
    MY = MY - MY.mean() + np.nanmean(DY)

    # display calibration (gaze_anim.py convention): sign + std match to the dot
    def disp(cal, ref):
        v = valid & np.isfinite(cal)
        rr = khz2d.corr(cal[v], ref[v])
        s = (1.0 if rr >= 0 else -1.0) * (ref[v].std() / (cal[v].std() + 1e-9))
        return s * (cal - cal[v].mean()) + ref[v].mean()
    RX = disp(ev["cal_x"], DX); RY = disp(ev["cal_y"], DY)
    RXm = np.where(valid, RX, np.nan); RYm = np.where(valid, RY, np.nan)
    print(f"[anim] {METHOD} @ {rate:.0f} Hz, {T} samples ({t[-1]-t[0]:.1f}s), "
          f"in-FOV {valid.mean()*100:.0f}%")
    print(f"[anim] r-vs-dot x={ev['r_dot_x']:.2f} y={ev['r_dot_y']:.2f}; "
          f"r-vs-tracker x={ev['r_trk_x']:.2f} y={ev['r_trk_y']:.2f}; OFF={R['off']:.2f}s")

    # center on the dot frame
    cx, cy = np.nanmean(DX), np.nanmean(DY)
    DXc, DYc, MXc, MYc = DX - cx, DY - cy, MX - cx, MY - cy
    RXc, RYc = RXm - cx, RYm - cy

    step = max(1, int(round(rate / VIDEO_FPS)))
    fps = rate / step
    tail = int(TAIL_S * rate)
    fr = np.arange(0, T, step)
    print(f"[anim] {len(fr)} video frames @ {fps:.1f} fps -> {len(fr)/fps:.1f}s")

    fig = plt.figure(figsize=(14, 7))
    axp = fig.add_subplot(1, 2, 1)
    axp.plot(DXc[::BG_DECIM], DYc[::BG_DECIM], color='C2', lw=.3, alpha=.25)
    axp.plot(RXc[::BG_DECIM], RYc[::BG_DECIM], color='C3', lw=.3, alpha=.18)
    lim = max(np.nanpercentile(np.abs(np.r_[DXc, DYc]), 99), 20) * 1.2
    axp.set_xlim(-lim, lim); axp.set_ylim(-lim, lim)
    axp.set_aspect('equal'); axp.grid(alpha=.3)
    axp.set_xlabel("horizontal (arcmin)"); axp.set_ylabel("vertical (arcmin)")
    axp.set_title(f"2D gaze @ {rate:.0f} Hz — particle filter on the line stream "
                  "(in-FOV only)")
    dtail, = axp.plot([], [], color='C2', lw=1.4, alpha=.5)
    mtail, = axp.plot([], [], color='C0', lw=1.2, alpha=.4)
    rtail, = axp.plot([], [], color='C3', lw=1.2, alpha=.6)
    ddot, = axp.plot([], [], '*', color='C2', ms=20, label='dot (target)')
    mdot, = axp.plot([], [], 'o', color='C0', ms=10, label='machine tracker (32.5 Hz)')
    rdot, = axp.plot([], [], 'o', color='C3', ms=9,
                     label=f'particle filter ({rate:.0f} Hz)')
    axp.legend(loc='upper right', fontsize=9)
    ttxt = axp.text(.03, .97, "", transform=axp.transAxes, va='top',
                    fontsize=11, family='monospace')

    def ts_panel(pos, D, M, Rm, ylab, title):
        ax = fig.add_subplot(2, 2, pos)
        ax.plot(t[::8], D[::8], color='C2', lw=1.0, alpha=.7, label='dot')
        ax.plot(t[::8], M[::8], color='C0', lw=.5, alpha=.6, label='machine')
        ax.plot(t[::4], Rm[::4], color='C3', lw=.4, label='recon')
        ylo = np.nanpercentile(np.r_[D, Rm[valid]], 0.5) - 15
        yhi = np.nanpercentile(np.r_[D, Rm[valid]], 99.5) + 15
        ax.fill_between(t[::8], ylo, yhi, where=~valid[::8],
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
    fig.suptitle(f"kHz 2D gaze from x-scan lines + 2D SLO — M4 particle filter @ "
                 f"{rate:.0f} Hz (strongest method, results/khz2d_methods.md)  |  "
                 f"in-FOV {valid.mean()*100:.0f}%", fontsize=11, y=.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def upd(i):
        j = fr[i]; a = max(0, j - tail)
        dtail.set_data(DXc[a:j+1:4], DYc[a:j+1:4])
        mtail.set_data(MXc[a:j+1:4], MYc[a:j+1:4])
        rtail.set_data(RXc[a:j+1:4], RYc[a:j+1:4])
        ddot.set_data([DXc[j]], [DYc[j]]); mdot.set_data([MXc[j]], [MYc[j]])
        if valid[j]:
            rdot.set_data([RXc[j]], [RYc[j]])
            s = f"err={np.hypot(RXc[j]-DXc[j], RYc[j]-DYc[j]):4.0f}'"
        else:
            rdot.set_data([], []); s = "OUT OF FOV"
        ttxt.set_text(f"t={t[j]:6.2f}s\n{s}")
        cvx.set_xdata([t[j], t[j]]); cvy.set_xdata([t[j], t[j]])
        return dtail, mtail, rtail, ddot, mdot, rdot, ttxt, cvx, cvy

    os.makedirs("results", exist_ok=True)
    ani = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000 / fps,
                                  blit=True)
    ani.save(OUT, writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    print(f"[anim] wrote {OUT} ({len(fr)} frames @ {fps:.1f} fps, "
          f"video {len(fr)/fps:.1f}s)")


if __name__ == "__main__":
    main()
