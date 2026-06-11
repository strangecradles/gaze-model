"""raster_track.py — strip-based 2D gaze tracker for the SLO raster (test1).

The decisive insight for high-rate gaze from a RASTER capture (vs the 1D
line-scan DPF): each raster frame is built column-by-column over ~68 ms, so a
STRIP of S consecutive slow-axis columns is a genuine 2D retinal patch acquired
at a sub-frame instant. 2D-registering each strip to a fixed reference mosaic
recovers absolute gaze at strip rate = (W_cols / S) * fps — hundreds of Hz to
the 11.8 kHz line rate — WITHOUT the 1D perp aliasing that forced the DPF
(a 2D patch match is unique). This is the Sheehy/Roorda TSLO method
(Biomed. Opt. Express 3(10):2611, 2012: 960 Hz @ 0.66' offline-1920 Hz) adapted
to the offline test1 capture.

Geometry (measured): test1 SLO_0 frame = (H=1000 FAST rows, W=808 SLOW cols),
1025 frames @ 14.633 fps (line rate 11823 Hz, 70 s).
  perp  = FAST axis (rows)  -> VERTICAL gaze   (weaker; right-eye FOV-limited)
  along = SLOW axis (cols)  -> HORIZONTAL gaze (strong axis)
A strip frame[:, c:c+S] is (1000, S); its acquisition time is
  t = f/fps + (c + S/2)/W / fps.

Two reference modes:
  'mosaic'      — coarse frame->frame0 offset positions a stitched composite
                  reference; each strip is fine-matched to a local mosaic ROI.
                  ABSOLUTE (drift-free); the substrate for accuracy.
  'incremental' — strip vs previous frame's same columns (relative, drifts);
                  kept only as a robustness cross-check.

No learned parameters. Outputs a per-strip record (t, perp_px, along_px, q,
contrast, infov, frame, strip). Scale (px->arcmin) is measured separately by
measure_scale() against the independent machine tracker.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

import data

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

# ---- default tracker constants (TSLO-style) ----
PAD = 64               # local 2D search half-window (px) around coarse prediction
Q_FOV = 0.35           # per-strip NCC floor for "in FOV / locked"
CONTRAST_FRAC = 0.5    # raw strip contrast floor (x median contrast)
MIN_COVER = 3          # min frames covering a mosaic pixel to trust it
BLINK_FRAC = 0.45      # frame mean-intensity floor (x median) -> blink/dropout


@dataclass
class RasterTrack:
    t: np.ndarray          # (N,) s, SLO clock
    perp_px: np.ndarray    # (N,) FAST-axis (vertical) gaze, mosaic px
    along_px: np.ndarray   # (N,) SLOW-axis (horizontal) gaze, mosaic px
    q: np.ndarray          # (N,) match quality (NCC peak)
    contrast: np.ndarray   # (N,) raw strip contrast (std)
    infov: np.ndarray      # (N,) bool — locked & in-FOV
    frame: np.ndarray      # (N,) int frame index
    strip: np.ndarray      # (N,) int strip index within frame
    S: int                 # cols per strip
    fps: float
    strip_hz: float        # (W//S) * fps
    W: int
    H: int
    ref_mode: str

    @property
    def n(self) -> int:
        return len(self.t)


def _nz(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return ((x - x.mean()) / (s if s > 0 else 1.0)).astype(np.float32)


def _parabolic(c: np.ndarray, k: int) -> float:
    if 0 < k < len(c) - 1:
        a, b, d = c[k - 1], c[k], c[k + 1]
        den = a - 2 * b + d
        if abs(den) > 1e-12:
            return float(k + 0.5 * (a - d) / den)
    return float(k)


def _load_frames(which: str, max_frames: Optional[int], rebuild: bool = False):
    """Debanded grayscale frames as a DISK MEMMAP; return (frames, fps, mean_int).

    The full test1 raster is ~3.3 GB; holding it in RAM OOM-kills. We decode once
    into a disk-backed float32 memmap (cache/frames_*.dat) plus a tiny sidecar
    (fps, mean_int, shape). The mosaic build and strip matching then random-access
    the memmap with the OS page cache, keeping resident RAM flat. Persisted across
    runs and reused across strip sizes within a rate sweep.
    """
    import cv2
    tag = "all" if max_frames is None else str(max_frames)
    dat = os.path.join(CACHE, f"frames_{which}_{tag}.dat")
    side = os.path.join(CACHE, f"frames_{which}_{tag}.npz")
    if os.path.exists(dat) and os.path.exists(side) and not rebuild:
        s = np.load(side)
        shape = tuple(int(x) for x in s["shape"])
        mm = np.memmap(dat, dtype=np.float32, mode="r", shape=shape)
        return mm, float(s["fps"]), s["mean_int"]

    cap = cv2.VideoCapture(data._slo_path(which))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = n_total if max_frames is None else min(n_total, max_frames)
    mm = np.memmap(dat, dtype=np.float32, mode="w+", shape=(n, H, W))
    mean_int = np.zeros(n, np.float32)
    i = 0
    while i < n:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean_int[i] = float(g.mean())
        mm[i] = data._deband(g)
        i += 1
    cap.release()
    if i < n:                                   # short read -> trim
        mm.flush(); del mm
        mm = np.memmap(dat, dtype=np.float32, mode="r+", shape=(n, H, W))[:i]
        n = i
    mm.flush()
    np.savez(side, fps=fps, mean_int=mean_int[:n], shape=np.array(mm.shape))
    mm = np.memmap(dat, dtype=np.float32, mode="r", shape=(n, H, W))
    return mm, float(fps), mean_int[:n]


_MOSAIC_CACHE: dict = {}


def _get_mosaic(which, max_frames, frames, mean_int):
    key = (which, max_frames)
    if key not in _MOSAIC_CACHE:
        _MOSAIC_CACHE[key] = build_mosaic(frames, mean_int)
    return _MOSAIC_CACHE[key]


def _stitch(frames, offs, good, py, px, H, W):
    Hc, Wc = H + 2 * py, W + 2 * px
    acc = np.zeros((Hc, Wc), np.float64)
    cov = np.zeros((Hc, Wc), np.int32)
    for i in range(len(frames)):
        if not good[i]:
            continue
        r0 = py + int(round(offs[i, 0])); c0 = px + int(round(offs[i, 1]))
        if r0 < 0 or c0 < 0 or r0 + H > Hc or c0 + W > Wc:
            continue
        acc[r0:r0 + H, c0:c0 + W] += frames[i]
        cov[r0:r0 + H, c0:c0 + W] += 1
    mosaic = np.where(cov > 0, acc / np.maximum(cov, 1), 0.0).astype(np.float32)
    return mosaic, cov


def build_mosaic(frames: np.ndarray, mean_int: np.ndarray, n_iter: int = 2):
    """Drift-free stitched composite reference (TSLO-style synthetic reference).

    Pass 0 seeds per-frame offsets by incremental frame->frame phase-correlation
    (cumulative). Passes 1.. RE-ANCHOR each frame to the *global* mosaic
    (phase-corr of the frame against the mosaic crop at its current estimate),
    killing the accumulation drift that an incremental-only cumsum builds up over
    70 s. Mirrors load_atlas's iterative register->average->repeat.

    Returns (mosaic (Hc,Wc) f32, cover int, offs (n,2) cumulative (cy,cx) rel
    frame0, pad (py,px), good (n,) bool).
    """
    n, H, W = frames.shape
    good = mean_int > BLINK_FRAC * np.median(mean_int)
    offs = np.zeros((n, 2), np.float32)
    prev = None
    for i in range(n):
        if prev is not None and good[i] and good[i - 1]:
            dy, dx = data._phase_corr2d(prev, np.asarray(frames[i]))
            offs[i] = offs[i - 1] + (dy, dx)
        elif i > 0:
            offs[i] = offs[i - 1]
        prev = np.asarray(frames[i]) if good[i] else prev
    offs -= offs[good].mean(0)

    def _pad(o):
        return (int(np.ceil(max(abs(o[:, 0].min()), abs(o[:, 0].max())))) + 6,
                int(np.ceil(max(abs(o[:, 1].min()), abs(o[:, 1].max())))) + 6)

    py, px = _pad(offs)
    mosaic, cov = _stitch(frames, offs, good, py, px, H, W)
    for _ in range(n_iter):
        Mn = _nz(np.where(cov >= 1, mosaic, 0.0))
        for i in range(n):
            if not good[i]:
                continue
            r0 = py + int(round(offs[i, 0])); c0 = px + int(round(offs[i, 1]))
            if r0 < 0 or c0 < 0 or r0 + H > Mn.shape[0] or c0 + W > Mn.shape[1]:
                continue
            crop = Mn[r0:r0 + H, c0:c0 + W]
            ddy, ddx = data._phase_corr2d(crop, np.asarray(frames[i]))
            if abs(ddy) < H // 2 and abs(ddx) < W // 2:        # reject wild jumps
                offs[i] = offs[i] + (ddy, ddx)
        offs[good] -= offs[good].mean(0)
        py, px = _pad(offs)
        mosaic, cov = _stitch(frames, offs, good, py, px, H, W)
    return mosaic, cov, offs, (py, px), good


def track(which: str = "test1", S: int = 8, ref_mode: str = "incremental",
          max_frames: Optional[int] = None, pad: int = PAD,
          rebuild: bool = False) -> RasterTrack:
    """Strip-track the raster at S cols/strip -> per-strip 2D gaze.

    strip_hz = (W // S) * fps. ref_mode:
      'incremental' (DEFAULT, proven) — each frame's strips are 2D-registered to
        the PREVIOUS frame (small inter-frame motion -> robust); per-frame motion
        = median strip shift, within-frame residual = strip - median. Relative
        (validated by detrended correlation), but tolerant of the large pursuit
        excursion + right-eye FOV loss that corrupt an absolute mosaic here.
      'mosaic' — absolute stitched reference (drift-free in principle, but
        degraded on this capture by FOV loss); kept as a cross-check.
    """
    import cv2
    tag = f"{which}_S{S}_{ref_mode}_{'all' if max_frames is None else max_frames}_p{pad}"
    cache = os.path.join(CACHE, f"rtrack_{tag}.npz")
    if os.path.exists(cache) and not rebuild:
        z = np.load(cache)
        return RasterTrack(z["t"], z["perp_px"], z["along_px"], z["q"], z["contrast"],
                           z["infov"].astype(bool), z["frame"], z["strip"], int(z["S"]),
                           float(z["fps"]), float(z["strip_hz"]), int(z["W"]), int(z["H"]),
                           str(z["ref_mode"]))

    frames, fps, mean_int = _load_frames(which, max_frames)
    n, H, W = frames.shape
    nstrip = W // S
    Tf = 1.0 / fps
    con_med = np.median([frames[f][:, c * S:(c + 1) * S].std()
                         for f in range(0, n, max(1, n // 20)) for c in range(nstrip)])

    T, PE, AL, Q, CON, FR, ST, COV = [], [], [], [], [], [], [], []

    if ref_mode == "mosaic":
        mosaic, cov, offs, (py, px), good = _get_mosaic(which, max_frames, frames, mean_int)
        Mn = _nz(np.where(cov >= MIN_COVER, mosaic, 0.0))
        covok = cov >= MIN_COVER
        Hc, Wc = Mn.shape
        for f in range(n):
            if not good[f]:
                continue
            cur = frames[f]
            ry = py + offs[f, 0]; rx = px + offs[f, 1]
            for s in range(nstrip):
                c = s * S
                strip = _nz(cur[:, c:c + S])               # (H, S)
                # coarse-predicted mosaic top-left for this strip
                top = int(round(ry)) - pad
                left = int(round(rx + c)) - pad
                t0 = max(0, top); l0 = max(0, left)
                t1 = min(Hc, top + H + 2 * pad); l1 = min(Wc, left + S + 2 * pad)
                if t1 - t0 < H or l1 - l0 < S:
                    continue
                roi = Mn[t0:t1, l0:l1]
                roiok = covok[t0:t1, l0:l1]
                r = cv2.matchTemplate(roi, strip, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                # sub-pixel refine on the NCC surface
                jx, jy = loc[0], loc[1]
                sub_x = _parabolic(r[jy, :], jx) if r.shape[1] > 2 else jx
                sub_y = _parabolic(r[:, jx], jy) if r.shape[0] > 2 else jy
                # absolute matched position in mosaic coords
                m_row = t0 + sub_y
                m_col = l0 + sub_x
                # coverage fraction of valid mosaic under the matched strip (FOV gate)
                cf = float(roiok[jy:jy + H, jx:jx + S].mean()) if roiok.size else 0.0
                perp = m_row - py                          # vs frame0 nominal
                along = m_col - (px + c)
                T.append(f * Tf + (c + S / 2) / W * Tf)
                PE.append(perp); AL.append(along); Q.append(mx)
                CON.append(float(cur[:, c:c + S].std())); COV.append(cf)
                FR.append(f); ST.append(s)
    elif ref_mode == "incremental":
        refp = None
        cumy = cumx = 0.0
        for f in range(n):
            cur = _nz(frames[f])
            if refp is None:
                refp = cv2.copyMakeBorder(cur, pad, pad, pad, pad,
                                          cv2.BORDER_CONSTANT, value=0)
                continue
            fy = np.zeros(nstrip); fx = np.zeros(nstrip); fq = np.zeros(nstrip)
            for s in range(nstrip):
                c = s * S
                strip = cur[:, c:c + S]
                reg = refp[:, c:c + S + 2 * pad]
                r = cv2.matchTemplate(reg, strip, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                fy[s] = _parabolic(r[:, loc[0]], loc[1]) - pad
                fx[s] = _parabolic(r[loc[1], :], loc[0]) - pad
                fq[s] = mx
            my, mxd = np.median(fy), np.median(fx)
            cumy += my; cumx += mxd
            for s in range(nstrip):
                c = s * S
                T.append(f * Tf + (c + S / 2) / W * Tf)
                PE.append(cumy + (fy[s] - my)); AL.append(cumx + (fx[s] - mxd))
                Q.append(fq[s]); CON.append(float(frames[f][:, c:c + S].std()))
                COV.append(1.0); FR.append(f); ST.append(s)
            refp = cv2.copyMakeBorder(cur, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=0)
    else:
        raise ValueError(f"unknown ref_mode {ref_mode!r}")

    t = np.asarray(T, np.float64)
    perp = np.asarray(PE, np.float32); along = np.asarray(AL, np.float32)
    q = np.asarray(Q, np.float32); con = np.asarray(CON, np.float32)
    cov = np.asarray(COV, np.float32)
    fr = np.asarray(FR, np.int32); st = np.asarray(ST, np.int32)
    infov = (q > Q_FOV) & (con > CONTRAST_FRAC * con_med) & (cov > 0.4)

    rt = RasterTrack(t, perp, along, q, con, infov, fr, st, S, fps,
                     (W // S) * fps, W, H, ref_mode)
    np.savez(cache, t=t, perp_px=perp, along_px=along, q=q, contrast=con,
             infov=infov, frame=fr, strip=st, S=S, fps=fps,
             strip_hz=rt.strip_hz, W=W, H=H, ref_mode=ref_mode)
    assert rt.perp_px.shape == rt.along_px.shape == rt.t.shape
    assert np.isfinite(rt.perp_px).all() and np.isfinite(rt.along_px).all()
    return rt


# ---------------------------------------------------------------------------
# Scale (px -> arcmin) measured against the INDEPENDENT machine tracker
# ---------------------------------------------------------------------------


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else np.nan


def _best_off(sig, t, ref, tref, lo=-1.0, hi=6.0, n=141):
    best = (0.0, 0.0)
    for o in np.linspace(lo, hi, n):
        c = _corr(sig, np.interp(t, tref + o, ref))
        if np.isfinite(c) and abs(c) > abs(best[1]):
            best = (float(o), float(c))
    return best


@dataclass
class RasterScale:
    arcmin_per_px_perp: float
    arcmin_per_px_along: float
    r_perp: float
    r_along: float
    off_s: float


def measure_scale(rt: RasterTrack, which: str = "test1") -> RasterScale:
    """Measure arcmin/px on BOTH raster axes by regression vs the independent
    ~32.5 Hz machine tracker (right_x*0.611 deg horizontal, right_y*0.461 deg
    vertical). Robust slope = sign-corrected std ratio at the best clock offset.
    """
    from scipy.ndimage import median_filter
    tr = data.load_tracker(which)
    ttr = tr.t_s - tr.t_s[0]

    def fill(x):
        x = np.asarray(x, float); i = np.arange(len(x)); m = np.isfinite(x)
        return np.interp(i, i[m], x[m]) if m.any() else x
    trx = fill(tr.right_x) * data.UNITS.tracker_deg_per_unit_x * 60.0   # arcmin
    tryv = fill(tr.right_y) * data.UNITS.tracker_deg_per_unit_y * 60.0
    fov = rt.infov
    H = median_filter(rt.along_px, 5)   # horizontal gaze = slow axis = along
    V = median_filter(rt.perp_px, 5)    # vertical gaze   = fast axis = perp
    # JOINT two-axis clock offset (a single-axis search aliases on this
    # per-axis-sinusoid stimulus; a true OFF aligns BOTH axes at once). The
    # mp4 creation_time leads the CSV by ~2.2-2.6 s -> search a plausible band.
    off, best = 2.2, -1.0
    for o in np.linspace(0.5, 4.0, 141):
        cx = _corr(H[fov], np.interp(rt.t[fov], ttr + o, trx))
        cy = _corr(V[fov], np.interp(rt.t[fov], ttr + o, tryv))
        s = (0.0 if not np.isfinite(cx) else abs(cx)) + (0.0 if not np.isfinite(cy) else abs(cy))
        if s > best:
            best, off = s, float(o)
    trxi = np.interp(rt.t, ttr + off, trx)
    tryi = np.interp(rt.t, ttr + off, tryv)
    rH = _corr(H[fov], trxi[fov]); rV = _corr(V[fov], tryi[fov])
    sH = (np.nanstd(trxi[fov]) / (np.nanstd(H[fov]) + 1e-9))
    sV = (np.nanstd(tryi[fov]) / (np.nanstd(V[fov]) + 1e-9))
    return RasterScale(float(sV), float(sH), float(rV), float(rH), float(off))


def _report(which="test1", S=16, max_frames=None):
    rt = track(which, S=S, max_frames=max_frames)
    sc = measure_scale(rt, which)
    print(f"[raster_track] {which} S={S} ref={rt.ref_mode}: {rt.n} strips over "
          f"{rt.t[-1]-rt.t[0]:.1f}s -> {rt.strip_hz:.0f} Hz (frame {rt.fps:.2f} Hz)")
    print(f"  match q median {np.median(rt.q):.2f}; in-FOV {rt.infov.mean()*100:.0f}%")
    print(f"  scale (vs 32.5Hz tracker, OFF={sc.off_s:.2f}s): "
          f"perp {sc.arcmin_per_px_perp:.3f}'/px (r={sc.r_perp:+.2f}), "
          f"along {sc.arcmin_per_px_along:.3f}'/px (r={sc.r_along:+.2f})")
    return rt, sc


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="strip-based 2D raster gaze tracker")
    p.add_argument("--which", default="test1")
    p.add_argument("--S", type=int, default=16)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.rebuild:
        track(a.which, S=a.S, max_frames=a.max_frames, rebuild=True)
    _report(a.which, S=a.S, max_frames=a.max_frames)
