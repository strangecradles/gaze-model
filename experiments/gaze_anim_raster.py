"""gaze_anim_raster.py — kHz 2D gaze tracker from the SLO full raster (test1).

The raster's fast axis is ~12 kHz and the slow axis ~14.6 Hz (one 808-sweep frame
per 68 ms). Each frame COLUMN is one fast-axis sweep; the slow axis steps between
columns, so a STRIP of consecutive columns is a genuine 2D retinal patch. Tracking
strip-by-strip (TSLO-style) recovers gaze at SUB-FRAME (hundreds of Hz to kHz)
resolution — far faster than the 14.6 Hz frame rate.

Method (validated: 483 Hz, horizontal r=-0.92 vs dot / -0.94 vs tracker):
  incremental rolling reference (frame f vs f-1), each strip 2D-registered to the
  previous frame's SAME columns (padded for a local 2D search via matchTemplate).
  Per-frame motion = median strip shift (robust); within-frame residual = each
  strip minus that median -> the sub-frame detail. Accumulate -> continuous 2D gaze.
    DX = slow-axis displacement  -> horizontal gaze (the strong axis)
    DY = fast-axis displacement  -> vertical gaze (weaker; fast-axis features)

FOV partitioning (no signal outside the SLO FOV): per-STRIP gate on the match
quality (q) AND raw strip contrast. Out-of-FOV / blink strips are masked so we
never track noise.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d, median_filter

import data
import calib
import eval_real as er

WHICH = "test1"
S = 20                 # columns (sweeps) per strip -> ~808/S strips/frame
PAD = 80               # local 2D search half-window (px)
Q_FOV = 0.35           # per-strip match-quality floor for in-FOV
CONTRAST_FRAC = 0.5    # raw strip contrast floor (x median)
CACHE = "cache/raster_khz_track.npz"
OUT = "results/gaze2d_anim_raster_khz.mp4"


def _nz(x):
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def track(rebuild=False):
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    cap = cv2.VideoCapture(data._slo_path(WHICH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(data._deband(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)))
    cap.release()
    frames = np.array(frames)
    n, H, W = frames.shape
    Tf = 1.0 / fps
    nstrip = W // S
    DY = []; DX = []; Q = []; CON = []; TT = []
    cumy = cumx = 0.0
    ref = _nz(frames[0])
    refp = cv2.copyMakeBorder(ref, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    rawprev = frames[0]
    for f in range(1, n):
        cur = _nz(frames[f]); raw = frames[f]
        fy = np.zeros(nstrip); fx = np.zeros(nstrip); fq = np.zeros(nstrip); fc = np.zeros(nstrip)
        for s in range(nstrip):
            c = s * S
            strip = cur[:, c:c + S]
            reg = refp[:, c:c + S + 2 * PAD]
            r = cv2.matchTemplate(reg, strip, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            fy[s] = loc[1] - PAD; fx[s] = loc[0] - PAD; fq[s] = mx
            fc[s] = raw[:, c:c + S].std()
        my = np.median(fy); mxd = np.median(fx)
        cumy += my; cumx += mxd
        for s in range(nstrip):
            DY.append(cumy + (fy[s] - my)); DX.append(cumx + (fx[s] - mxd))
            Q.append(fq[s]); CON.append(fc[s]); TT.append(f * Tf + (s * S + S / 2) / W * Tf)
        ref = cur; refp = cv2.copyMakeBorder(ref, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)
    out = dict(DY=np.array(DY), DX=np.array(DX), Q=np.array(Q), CON=np.array(CON),
               T=np.array(TT), fps=np.array(fps), strip_hz=np.array(nstrip * fps), F0=np.array(1.0 / fps))
    np.savez(CACHE, **out)
    return out


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a * b).sum() / d if d > 0 else np.nan


def _best_off(sig, t, ref, tref, valid):
    best = (0.0, 0.0)
    for o in np.linspace(-1, 6, 141):
        c = _corr(sig[valid], np.interp(t[valid], tref + o, ref))
        if np.isfinite(c) and abs(c) > abs(best[1]):
            best = (float(o), float(c))
    return best


def main():
    tk = track()
    DY, DX, Q, CON, T = tk["DY"], tk["DX"], tk["Q"], tk["CON"], tk["T"]
    fps = float(tk["fps"]); strip_hz = float(tk["strip_hz"])
    T = T + 1.0 / fps  # SLO clock from frame 1
    fov = (Q > Q_FOV) & (CON > CONTRAST_FRAC * np.median(CON))
    print(f"[raster] {len(DY)} strips over {T[-1]-T[0]:.1f}s -> {strip_hz:.0f} Hz "
          f"(frame {fps:.1f}Hz); match med {np.median(Q):.2f}; in-FOV {fov.mean()*100:.0f}%")

    # arcmin + light smoothing + slow-drift removal (keep the 0.2 Hz pursuit band)
    def proc(v):
        v = calib.rows_to_arcmin(v.astype(float))
        v = gaussian_filter1d(median_filter(v, 7), max(1.0, strip_hz * 0.01))
        return v - gaussian_filter1d(v, strip_hz / (2 * np.pi * 0.05))
    H = proc(DX); V = proc(DY)

    st = data.load_stimulus("pursuit"); dotX = st.x_deg * 60; dotY = st.y_deg * 60
    tr = data.load_tracker(WHICH); ttr = tr.t_s - tr.t_s[0]
    def fill(x):
        x = np.asarray(x, float); i = np.arange(len(x)); m = np.isfinite(x)
        return np.interp(i, i[m], x[m])
    trX = fill(tr.right_x) * er.TRK_DEG_X * 60; trY = fill(tr.right_y) * er.TRK_DEG_Y * 60

    offH, rH = _best_off(H, T, dotX, st.t_s, fov)
    offV, rV = _best_off(V, T, dotY, st.t_s, fov)
    OFF = offH
    DXd = np.interp(T, st.t_s + OFF, dotX); DYd = np.interp(T, st.t_s + OFF, dotY)
    MXd = np.interp(T, ttr + OFF, trX); MYd = np.interp(T, ttr + OFF, trY)
    rHt = _corr(H[fov], MXd[fov]); rVt = _corr(V[fov], MYd[fov])
    print(f"[raster] horiz(DX) vs dot r={rH:+.2f} (vs tracker {rHt:+.2f}); "
          f"vert(DY) vs dot r={rV:+.2f} (vs tracker {rVt:+.2f}); OFF={OFF:.2f}s")

    def cal(rec, ref):
        v = fov & np.isfinite(rec) & np.isfinite(ref)
        r = _corr(rec[v], ref[v]); sr = rec[v].std(); sref = ref[v].std()
        return (1 if r >= 0 else -1) * (sref / sr if sr > 0 else 1) * (rec - rec[v].mean())
    RX = cal(H, DXd); RY = cal(V, DYd)
    RXm = np.where(fov, RX, np.nan); RYm = np.where(fov, RY, np.nan)
    cx, cy = np.nanmean(DXd), np.nanmean(DYd)
    DXd, DYd, MXd, MYd, RXm, RYm = DXd-cx, DYd-cy, MXd-cx, MYd-cy, RXm-cx, RYm-cy

    # real-time animation
    step = max(1, int(round(strip_hz / 30))); fpsv = strip_hz / step
    tail = int(0.5 * strip_hz / step); fr = np.arange(0, len(T), step)
    fig = plt.figure(figsize=(14, 7))
    axp = fig.add_subplot(1, 2, 1)
    axp.plot(DXd, DYd, 'C2', lw=.3, alpha=.25); axp.plot(RXm, RYm, 'C3', lw=.2, alpha=.15)
    lim = max(np.nanpercentile(np.abs(np.r_[DXd, DYd]), 99), 20) * 1.2
    axp.set_xlim(-lim, lim); axp.set_ylim(-lim, lim); axp.set_aspect('equal'); axp.grid(alpha=.3)
    axp.set_xlabel("horizontal (')"); axp.set_ylabel("vertical (')")
    axp.set_title(f"2D gaze @ {strip_hz:.0f} Hz from the raster (strip tracking)")
    dt_, = axp.plot([], [], 'C2', lw=1.4, alpha=.5); mt_, = axp.plot([], [], 'C0', lw=1.0, alpha=.35)
    rt_, = axp.plot([], [], 'C3', lw=1.2, alpha=.6)
    dd, = axp.plot([], [], '*', color='C2', ms=20, label='dot (target)')
    md, = axp.plot([], [], 'o', color='C0', ms=10, label='machine (14.6Hz)')
    rd, = axp.plot([], [], 'o', color='C3', ms=8, label=f'raster recon ({strip_hz:.0f}Hz)')
    axp.legend(loc='upper right', fontsize=9)
    tt = axp.text(.03, .97, "", transform=axp.transAxes, va='top', fontsize=11, family='monospace')

    def panel(pos, D, M, R, yl, title):
        ax = fig.add_subplot(2, 2, pos)
        ax.plot(T, D, 'C2', lw=1.0, alpha=.7, label='dot'); ax.plot(T, M, 'C0', lw=.5, alpha=.5, label='machine')
        ax.plot(T, R, 'C3', lw=.4, label='recon')
        i = 0
        while i < len(T):
            if not fov[i]:
                j = i
                while j < len(T) and not fov[j]:
                    j += 1
                ax.axvspan(T[i], T[min(j, len(T)-1)], color='0.85', alpha=.5, lw=0); i = j
            else:
                i += 1
        ax.set_ylabel(yl); ax.set_title(title); ax.grid(alpha=.3)
        return ax, ax.axvline(T[0], color='k', lw=1)
    axx, cvx = panel(2, DXd, MXd, RXm, "horiz (')", f"Horizontal (slow axis)  recon-vs-dot r={rH:+.2f}")
    axx.legend(fontsize=7, loc='upper right')
    axy, cvy = panel(4, DYd, MYd, RYm, "vert (')", f"Vertical (fast axis)  recon-vs-dot r={rV:+.2f}")
    axy.set_xlabel("time (s)")
    fig.suptitle(f"kHz 2D gaze from SLO raster: {strip_hz:.0f} Hz (frame rate {fps:.1f} Hz)  |  "
                 f"in-FOV {fov.mean()*100:.0f}%  |  horiz r={rH:+.2f}, vert r={rV:+.2f}", fontsize=11, y=.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def upd(i):
        j = fr[i]; a = max(0, j - tail)
        dt_.set_data(DXd[a:j+1], DYd[a:j+1]); mt_.set_data(MXd[a:j+1], MYd[a:j+1]); rt_.set_data(RXm[a:j+1], RYm[a:j+1])
        dd.set_data([DXd[j]], [DYd[j]]); md.set_data([MXd[j]], [MYd[j]])
        if fov[j]:
            rd.set_data([RXm[j]], [RYm[j]]); s = f"err={np.hypot(RXm[j]-DXd[j], RYm[j]-DYd[j]):4.0f}'"
        else:
            rd.set_data([], []); s = "OUT OF FOV"
        tt.set_text(f"t={T[j]:6.2f}s\n{s}")
        cvx.set_xdata([T[j], T[j]]); cvy.set_xdata([T[j], T[j]])
        return dt_, mt_, rt_, dd, md, rd, tt, cvx, cvy

    os.makedirs("results", exist_ok=True)
    ani = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000/fpsv, blit=True)
    ani.save(OUT, writer=animation.FFMpegWriter(fps=fpsv, bitrate=3000))
    print(f"[raster] wrote {OUT} ({len(fr)} frames @ {fpsv:.1f}fps, video {len(fr)/fpsv:.1f}s)")


if __name__ == "__main__":
    main()
