"""gaze_anim_test1.py — gaze-vs-dot animation using a SAME-SESSION (test1) atlas.

Diagnosis recap: our particle filter could not track test2 against the normal/
atlas because that atlas is a different capture/zoom (cross-scale) — the perp
appearance never locked (NCC ~0.24). test1 is the pursuit RASTER recorded in the
SAME session as the test2 x-scan, so a test1-built mosaic is NATIVE SCALE. test2
lines now match it at q~0.86 (vs 0.24 cross-scale). The localized perp position
still tracks the dot only modestly (|r|~0.25-0.38) — the documented single-line
aliasing wall — so this is HONEST degree-scale tracking, not sub-0.1deg.

This is the physics appearance-match (the DPF's observation model) on the
same-session atlas: perp = coarse-band atlas-row localization (light temporal
smoothing for continuity), along = trusted within-line shift. Gated to in-FOV
samples; calibrated into the dot's arcmin frame (sign + amplitude match).
"""
from __future__ import annotations
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d, median_filter

import data

WHICH = "test2"
MAX_SWEEPS = 560_000        # ~47 s
BLOCK = 120                 # super-sweep averaging -> ~100 Hz, q~0.86
OUT = "results/gaze2d_anim_test1atlas.mp4"
CACHE = "cache/test1atlas_track.npz"


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a * b).sum() / d if d > 0 else np.nan


def _best_off(sig, t, ref, t_ref):
    best = (0.0, 0.0)
    for o in np.linspace(-2, 6, 161):
        c = _corr(sig, np.interp(t, t_ref + o, ref))
        if np.isfinite(c) and abs(c) > abs(best[1]):
            best = (float(o), float(c))
    return best


def _track(rebuild=False):
    import os
    if os.path.exists(CACHE) and not rebuild:
        z = np.load(CACHE)
        return z["perp"], z["along"], z["q"], z["contrast"], float(z["fs"])
    atlas = np.load("cache/atlas_test1_mosaic.npy").T          # native-scale same-session
    Ac = gaussian_filter1d(atlas, 8, axis=1)
    Ac = ((Ac - Ac.mean()) / (Ac.std() + 1e-9)).astype(np.float32)
    Hr = Ac.shape[0]
    ls = data.load_line_scan(WHICH, max_sweeps=MAX_SWEEPS)
    fs = ls.line_rate_hz / BLOCK
    sw = ls.sweeps - ls.sweeps.mean(0, keepdims=True)
    nb = len(sw) // BLOCK
    blk = sw[:nb * BLOCK].reshape(nb, BLOCK, -1).mean(1)
    raw = ls.sweeps[:nb * BLOCK].reshape(nb, BLOCK, -1).mean(1)
    contrast = raw.std(1)
    perp = np.zeros(nb); q = np.zeros(nb)
    for i, L in enumerate(blk):
        Lc = gaussian_filter1d(L, 8)
        Lc = ((Lc - Lc.mean()) / (Lc.std() + 1e-9)).astype(np.float32)
        r = cv2.matchTemplate(Ac, Lc.reshape(1, -1), cv2.TM_CCOEFF_NORMED).max(1)
        perp[i] = r.argmax(); q[i] = r.max()
    along = data.along_shift(WHICH, max_sweeps=MAX_SWEEPS)[:nb * BLOCK].reshape(nb, BLOCK).mean(1)
    np.savez(CACHE, perp=perp, along=along, q=q, contrast=contrast, fs=fs)
    return perp, along, q, contrast, fs


def main():
    import calib
    perp, along, q, contrast, fs = _track()
    T = len(perp); t = np.arange(T) / fs
    # in-FOV: structured line (contrast) AND a real atlas match (q)
    fov = (contrast > 0.5 * np.median(contrast)) & (q > 0.5)
    # light temporal smoothing of the aliased perp argmax
    perp_s = gaussian_filter1d(median_filter(perp, 7), max(1.0, fs * 0.15))
    along_s = gaussian_filter1d(median_filter(along, 7), max(1.0, fs * 0.15))
    perp_a = calib.rows_to_arcmin(perp_s); along_a = calib.rows_to_arcmin(along_s)
    # remove slow drift below the 0.2 Hz pursuit
    perp_a = perp_a - gaussian_filter1d(perp_a, fs / (2 * np.pi * 0.06))
    along_a = along_a - gaussian_filter1d(along_a, fs / (2 * np.pi * 0.06))

    st = data.load_stimulus("pursuit")
    dotX = st.x_deg * 60; dotY = st.y_deg * 60
    import eval_real as er
    tr = data.load_tracker(WHICH); ttr = tr.t_s - tr.t_s[0]
    def fill(x):
        x = np.asarray(x, float); i = np.arange(len(x)); m = np.isfinite(x)
        return np.interp(i, i[m], x[m])
    trX = fill(tr.right_x) * er.TRK_DEG_X * 60; trY = fill(tr.right_y) * er.TRK_DEG_Y * 60

    # assign each channel to the dot axis it best tracks (at its own best OFF)
    op = _best_off(perp_a[fov], t[fov], dotX, st.t_s); oa = _best_off(along_a[fov], t[fov], dotY, st.t_s)
    # try both pairings, keep the stronger
    pX = abs(_best_off(perp_a[fov], t[fov], dotX, st.t_s)[1]) + abs(_best_off(along_a[fov], t[fov], dotY, st.t_s)[1])
    pY = abs(_best_off(perp_a[fov], t[fov], dotY, st.t_s)[1]) + abs(_best_off(along_a[fov], t[fov], dotX, st.t_s)[1])
    if pX >= pY:
        recH, recV, hlab, vlab = perp_a, along_a, "perp(appearance)", "along(shift)"
        refH, refV = dotX, dotY
    else:
        recH, recV, hlab, vlab = along_a, perp_a, "along(shift)", "perp(appearance)"
        refH, refV = dotX, dotY
    offH, rH = _best_off(recH[fov], t[fov], refH, st.t_s)
    offV, rV = _best_off(recV[fov], t[fov], refV, st.t_s)
    OFF = offH
    DX = np.interp(t, st.t_s + OFF, dotX); DY = np.interp(t, st.t_s + OFF, dotY)
    MX = np.interp(t, ttr + OFF, trX); MY = np.interp(t, ttr + OFF, trY)

    def calibd(rec, ref):
        v = fov & np.isfinite(rec) & np.isfinite(ref)
        r = _corr(rec[v], ref[v]); sr = rec[v].std(); sref = ref[v].std()
        return (1 if r >= 0 else -1) * (sref / sr if sr > 0 else 1) * (rec - rec[v].mean()), r
    RX, rxr = calibd(recH, DX); RY, ryr = calibd(recV, DY)
    print(f"[t1anim] in-FOV {fov.mean()*100:.0f}%  q med {np.median(q):.2f}")
    print(f"[t1anim] recon-vs-dot  horiz({hlab}) r={rxr:+.2f}  vert({vlab}) r={ryr:+.2f}  OFF={OFF:.2f}s")
    RXm = np.where(fov, RX, np.nan); RYm = np.where(fov, RY, np.nan)
    cx, cy = np.nanmean(DX), np.nanmean(DY)
    DX, DY, MX, MY, RXm, RYm = DX-cx, DY-cy, MX-cx, MY-cy, RXm-cx, RYm-cy

    step = max(1, int(round(fs / 30))); fps = fs / step
    tail = int(0.8 * fs / step); fr = np.arange(0, T, step)
    fig = plt.figure(figsize=(14, 7))
    axp = fig.add_subplot(1, 2, 1)
    axp.plot(DX, DY, 'C2', lw=.3, alpha=.25); axp.plot(RXm, RYm, 'C3', lw=.3, alpha=.2)
    lim = max(np.nanpercentile(np.abs(np.r_[DX, DY]), 99), 20) * 1.2
    axp.set_xlim(-lim, lim); axp.set_ylim(-lim, lim); axp.set_aspect('equal'); axp.grid(alpha=.3)
    axp.set_xlabel("horizontal (')"); axp.set_ylabel("vertical (')")
    axp.set_title("2D gaze vs dot — same-session (test1) atlas")
    dt_, = axp.plot([], [], 'C2', lw=1.4, alpha=.5); mt_, = axp.plot([], [], 'C0', lw=1.1, alpha=.4)
    rt_, = axp.plot([], [], 'C3', lw=1.4, alpha=.6)
    dd, = axp.plot([], [], '*', color='C2', ms=20, label='dot (target)')
    md, = axp.plot([], [], 'o', color='C0', ms=10, label='machine tracker')
    rd, = axp.plot([], [], 'o', color='C3', ms=9, label='our recon (map-match)')
    axp.legend(loc='upper right', fontsize=10)
    tt = axp.text(.03, .97, "", transform=axp.transAxes, va='top', fontsize=11, family='monospace')

    def panel(pos, D, M, R, yl, title):
        ax = fig.add_subplot(2, 2, pos)
        ax.plot(t, D, 'C2', lw=1.0, alpha=.7, label='dot'); ax.plot(t, M, 'C0', lw=.5, alpha=.6, label='machine')
        ax.plot(t, R, 'C3', lw=.7, label='recon')
        ax.set_ylabel(yl); ax.set_title(title); ax.grid(alpha=.3)
        return ax, ax.axvline(t[0], color='k', lw=1)
    axx, cvx = panel(2, DX, MX, RXm, "horiz (')", f"Horizontal ({hlab})  recon-vs-dot r={rxr:+.2f}")
    axx.legend(fontsize=7, loc='upper right')
    axy, cvy = panel(4, DY, MY, RYm, "vert (')", f"Vertical ({vlab})  recon-vs-dot r={ryr:+.2f}")
    axy.set_xlabel("time (s)")
    fig.suptitle(f"Same-session atlas (q~{np.median(q):.2f})  |  in-FOV {fov.mean()*100:.0f}%  |  "
                 f"degree-scale partial tracking (single-line aliasing wall)", fontsize=11, y=.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    def upd(i):
        j = fr[i]; a = max(0, j - tail)
        dt_.set_data(DX[a:j+1], DY[a:j+1]); mt_.set_data(MX[a:j+1], MY[a:j+1]); rt_.set_data(RXm[a:j+1], RYm[a:j+1])
        dd.set_data([DX[j]], [DY[j]]); md.set_data([MX[j]], [MY[j]])
        if fov[j]:
            rd.set_data([RXm[j]], [RYm[j]]); s = f"recon-dot ={np.hypot(RXm[j]-DX[j], RYm[j]-DY[j]):4.0f}'"
        else:
            rd.set_data([], []); s = "OUT OF FOV"
        tt.set_text(f"t={t[j]:6.2f}s\n{s}")
        cvx.set_xdata([t[j], t[j]]); cvy.set_xdata([t[j], t[j]])
        return dt_, mt_, rt_, dd, md, rd, tt, cvx, cvy

    import os
    os.makedirs("results", exist_ok=True)
    ani = animation.FuncAnimation(fig, upd, frames=len(fr), interval=1000/fps, blit=True)
    ani.save(OUT, writer=animation.FFMpegWriter(fps=fps, bitrate=2600))
    print(f"[t1anim] wrote {OUT} ({len(fr)} frames @ {fps:.1f}fps)")


if __name__ == "__main__":
    main()
