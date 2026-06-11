"""gaze_anim.py — animate OUR particle-filter gaze trajectory against the moving
dot the subject followed during acquisition (test2 pursuit x-scan).

Style follows saccade-tracking/gaze2d_anim_full.py: a 2D plane (dot star + recon
dot + fading tails) plus horizontal/vertical time-series with a moving cursor, in
a dot-referenced arcmin frame, played in real time.

IN-FOV GATING (the requirement): we track only where the eye was inside the SLO
field of view. Out-of-FOV / blink / noise samples are detected by LOW RAW LINE
CONTRAST (the SLO line goes dark/flat when no retina is imaged) combined with LOW
atlas-match quality (max_ncc) — those frames are masked out (recon hidden, spans
shaded), so we do not draw a "track" through noise.

Recon is in atlas units (perp rows, along cols); it is calibrated into the dot's
arcmin frame by a per-axis affine fit on the in-FOV samples (standard gaze
calibration to the independent dot reference — the same train-on-dot step the
sibling used). The recon-vs-dot correlation is reported on the figure.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import median_filter, gaussian_filter1d

import data
import calib
import eval_real as er

# ---- config ----
WINDOW_SWEEPS = 660_000     # ~55 s of test2 @ ~12 kHz (covers the pursuit, not just the sync intro)
EFF_RATE = 300.0            # effective filter rate (Hz) — smooth, well above the dot's 0.2 Hz
DUR_CAP_S = 54.0
NCC_FOV = 0.20             # atlas-match quality floor for "in FOV"
CONTRAST_FRAC = 0.5        # raw line contrast must exceed this * median
MIN_VALID_S = 0.20         # drop in-FOV islands shorter than this
MAX_GAP_S = 0.12           # bridge out-of-FOV gaps shorter than this
OUT = "results/gaze2d_anim.mp4"


def _raw_contrast(block: int, T: int) -> np.ndarray:
    """Per-effective-step contrast of the RAW (pre-deband) line — the FOV cue.

    Streamed from the mp4 (memory-light): concatenate sweeps, block-average raw
    lines, take the along-axis std. A dark/flat out-of-FOV (or blink) line has
    low contrast; a structured retinal line is high.
    """
    import cv2
    path = data._slo_path(er.WHICH)
    cap = cv2.VideoCapture(path)
    buf = []
    out = []
    got = 0
    while len(out) < T:
        ok, f = cap.read()
        if not ok:
            break
        sw = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32).T  # (n_sweeps, along)
        buf.append(sw); got += sw.shape[0]
        while got >= block and len(out) < T:
            chunk = np.concatenate(buf, 0)
            out.append(float(chunk[:block].mean(0).std()))
            rem = chunk[block:]
            buf = [rem] if len(rem) else []
            got = rem.shape[0]
    cap.release()
    arr = np.array(out[:T], np.float64)
    if len(arr) < T:                       # pad tail if the stream ran short
        arr = np.concatenate([arr, np.full(T - len(arr), arr[-1] if len(arr) else 1.0)])
    return arr


def _fov_mask(contrast: np.ndarray, max_ncc: np.ndarray, rate: float) -> np.ndarray:
    m = (contrast > CONTRAST_FRAC * np.median(contrast)) & (max_ncc > NCC_FOV)
    # bridge short gaps, then drop short islands (runs in samples)
    def runs(mask, val):
        out = []
        i = 0
        while i < len(mask):
            if mask[i] == val:
                j = i
                while j < len(mask) and mask[j] == val:
                    j += 1
                out.append((i, j))
                i = j
            else:
                i += 1
        return out
    gap = int(MAX_GAP_S * rate)
    for a, b in runs(m, False):
        if b - a <= gap:
            m[a:b] = True
    isl = int(MIN_VALID_S * rate)
    for a, b in runs(m, True):
        if b - a < isl:
            m[a:b] = False
    return m


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else 0.0


def _calib_disp(rec_slow, ref, valid):
    """Display-calibrate rec_slow into the ref (dot) frame: SIGN-align and
    AMPLITUDE (std) match — NOT a least-squares fit (which collapses to ~flat when
    the correlation is low). This preserves the recon's true relative motion so the
    viewer sees what the filter actually estimated. Returns centered mapped + corr."""
    v = valid & np.isfinite(rec_slow) & np.isfinite(ref)
    r = _corr(rec_slow[v], ref[v])
    sr = rec_slow[v].std(); sref = ref[v].std()
    scale = (1.0 if r >= 0 else -1.0) * (sref / sr if sr > 0 else 1.0)
    return scale * (rec_slow - rec_slow[v].mean()), r


def main():
    run = er.run_real_filter(EFF_RATE, DUR_CAP_S, window_sweeps=WINDOW_SWEEPS)
    rate = run.rate
    T = len(run.est_perp)
    t_est = np.arange(T) / rate
    line_rate = data.line_scan_meta(er.WHICH)["line_rate_hz"]
    block = max(1, int(round(line_rate / rate)))

    # --- in-FOV mask ---
    contrast = _raw_contrast(block, T)
    fov = _fov_mask(contrast, run.max_ncc, rate)
    print(f"[anim] T={T} @ {rate:.0f}Hz ({T/rate:.1f}s); in-FOV {fov.mean()*100:.0f}%")

    # --- recon SLOW component (atlas units -> arcmin) ---
    # The trustworthy real-data component is the slow trajectory (G15: cross-scale
    # makes the instantaneous fine track noisy). Remove drift slower than the 0.2 Hz
    # pursuit (the along axis is a drift-laden cumsum) and low-pass to the gaze band.
    SLOW_HZ = 1.5
    def _slow(x):
        x = calib.rows_to_arcmin(x.astype(float))
        x = x - gaussian_filter1d(x, rate / (2 * np.pi * 0.06))   # kill <0.06 Hz drift
        return gaussian_filter1d(median_filter(x, 5), max(1.0, rate / (2 * np.pi * SLOW_HZ)))
    rec_perp = _slow(run.est_perp); rec_along = _slow(run.est_along)

    # --- dot (what the subject followed) + machine tracker, in degrees->arcmin ---
    st = data.load_stimulus("pursuit")
    t_dot = st.t_s
    dot_x = st.x_deg * 60.0; dot_y = st.y_deg * 60.0
    tr = data.load_tracker(er.WHICH)
    t_tr = tr.t_s - tr.t_s[0]
    trk_x = tr.right_x * er.TRK_DEG_X * 60.0
    trk_y = tr.right_y * er.TRK_DEG_Y * 60.0

    # --- align recon to the playback clock: OFF maximizing combined |corr| vs
    #     BOTH the dot and the (stronger) machine tracker on in-FOV samples ---
    va = fov & np.isfinite(rec_along) & np.isfinite(rec_perp)
    best = (2.2, -9.0)
    for o in np.linspace(0.0, 6.0, 121):
        dx = np.interp(t_est, t_dot + o, dot_x); dy = np.interp(t_est, t_dot + o, dot_y)
        mx = np.interp(t_est, t_tr + o, trk_x); my = np.interp(t_est, t_tr + o, trk_y)
        s = (abs(_corr(rec_along[va], dx[va])) + abs(_corr(rec_perp[va], dy[va])) +
             abs(_corr(rec_along[va], mx[va])) + abs(_corr(rec_perp[va], my[va])))
        if s > best[1]:
            best = (float(o), s)
    off = best[0]
    DX = np.interp(t_est, t_dot + off, dot_x); DY = np.interp(t_est, t_dot + off, dot_y)
    MX = np.interp(t_est, t_tr + off, trk_x); MY = np.interp(t_est, t_tr + off, trk_y)
    print(f"[anim] OFF={off:.2f}s (SLO leads playback)")

    # --- display-calibrate recon into the dot frame (sign + amplitude match) ---
    RX, rx_r = _calib_disp(rec_along, DX, fov)   # horizontal
    RY, ry_r = _calib_disp(rec_perp, DY, fov)    # vertical
    rx_rt = _corr(rec_along[va], MX[va]); ry_rt = _corr(rec_perp[va], MY[va])
    print(f"[anim] recon-vs-dot horiz r={rx_r:.2f} vert r={ry_r:.2f}; "
          f"recon-vs-tracker horiz r={rx_rt:.2f} vert r={ry_rt:.2f}")

    # mask recon outside FOV so we never draw a track through noise
    RXm = np.where(fov, RX, np.nan); RYm = np.where(fov, RY, np.nan)

    # center on the dot frame
    cx, cy = np.nanmean(DX), np.nanmean(DY)
    DX, DY, MX, MY, RXm, RYm = DX-cx, DY-cy, MX-cx, MY-cy, RXm-cx, RYm-cy

    # --- animation (real time), reference style ---
    step = max(1, int(round(rate / 30)))
    fps = rate / step
    tail = int(0.8 * rate / step)
    fr = np.arange(0, T, step)
    print(f"[anim] {len(fr)} frames, fps={fps:.1f}, video={len(fr)/fps:.1f}s")

    fig = plt.figure(figsize=(14, 7))
    axp = fig.add_subplot(1, 2, 1)
    axp.plot(DX, DY, color='C2', lw=.3, alpha=.25)
    axp.plot(RXm, RYm, color='C3', lw=.3, alpha=.2)
    lim = max(np.nanpercentile(np.abs(np.r_[DX, DY, RXm[fov], RYm[fov]]), 99), 20) * 1.15
    axp.set_xlim(-lim, lim); axp.set_ylim(-lim, lim); axp.set_aspect('equal'); axp.grid(alpha=.3)
    axp.set_xlabel("horizontal (arcmin)"); axp.set_ylabel("vertical (arcmin)")
    axp.set_title("2D gaze vs pursuit dot (in-FOV only)")
    dtail, = axp.plot([], [], color='C2', lw=1.4, alpha=.5)
    mtail, = axp.plot([], [], color='C0', lw=1.2, alpha=.4)
    rtail, = axp.plot([], [], color='C3', lw=1.4, alpha=.6)
    ddot, = axp.plot([], [], '*', color='C2', ms=20, label='dot (target)')
    mdot, = axp.plot([], [], 'o', color='C0', ms=10, label='machine tracker')
    rdot, = axp.plot([], [], 'o', color='C3', ms=9, label='our filter (recon)')
    axp.legend(loc='upper right', fontsize=10)
    ttxt = axp.text(.03, .97, "", transform=axp.transAxes, va='top', fontsize=11, family='monospace')

    def ts_panel(pos, D, M, R, ylab, title):
        ax = fig.add_subplot(2, 2, pos)
        ax.plot(t_est, D, color='C2', lw=1.0, alpha=.7, label='dot')
        ax.plot(t_est, M, color='C0', lw=.5, alpha=.6, label='machine')
        ax.plot(t_est, R, color='C3', lw=.6, label='recon')
        # shade out-of-FOV spans
        inv = ~fov
        i = 0
        while i < T:
            if inv[i]:
                j = i
                while j < T and inv[j]:
                    j += 1
                ax.axvspan(t_est[i], t_est[min(j, T-1)], color='0.85', alpha=.5, lw=0)
                i = j
            else:
                i += 1
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=.3)
        cv = ax.axvline(t_est[0], color='k', lw=1)
        return ax, cv
    axx, cvx = ts_panel(2, DX, MX, RXm, "horiz (')",
                        f"Horizontal  recon-vs-dot r={rx_r:.2f}  (recon-vs-tracker r={rx_rt:.2f})")
    axx.legend(fontsize=7, loc='upper right')
    axy, cvy = ts_panel(4, DY, MY, RYm, "vert (')",
                        f"Vertical  recon-vs-dot r={ry_r:.2f}  (recon-vs-tracker r={ry_rt:.2f})")
    axy.set_xlabel("time (s)")
    fig.suptitle(f"Our particle-filter gaze vs pursuit dot  |  in-FOV {fov.mean()*100:.0f}%  |  "
                 f"slow component, sign+amplitude-matched  |  cross-scale-limited (G15)",
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def upd(i):
        j = fr[i]; a = max(0, j - tail)
        dtail.set_data(DX[a:j+1], DY[a:j+1])
        mtail.set_data(MX[a:j+1], MY[a:j+1])
        rtail.set_data(RXm[a:j+1], RYm[a:j+1])
        ddot.set_data([DX[j]], [DY[j]]); mdot.set_data([MX[j]], [MY[j]])
        if fov[j]:
            rdot.set_data([RXm[j]], [RYm[j]])
            e = np.hypot(RXm[j]-DX[j], RYm[j]-DY[j])
            status = f"recon-dot ={e:4.0f}'"
        else:
            rdot.set_data([], [])
            status = "OUT OF FOV (no track)"
        ttxt.set_text(f"t={t_est[j]:6.2f}s\n{status}")
        cvx.set_xdata([t_est[j], t_est[j]]); cvy.set_xdata([t_est[j], t_est[j]])
        return dtail, mtail, rtail, ddot, mdot, rdot, ttxt, cvx, cvy

    ani = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000/fps, blit=True)
    import os
    os.makedirs("results", exist_ok=True)
    ani.save(OUT, writer=animation.FFMpegWriter(fps=fps, bitrate=2600))
    print(f"[anim] wrote {OUT}")


if __name__ == "__main__":
    main()
