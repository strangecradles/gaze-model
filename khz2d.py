"""khz2d.py — shared harness for the kHz 2D gaze multi-method study.

GOAL: reconstruct 2D gaze at kHz output rate by fusing the ~12 kHz fast-axis
line stream with the ~15 Hz (nominal 20 Hz) 2D SLO frames.

TESTBED A = the test1 pursuit RASTER. Its own columns ARE the kHz x-scan
(each column = one ~85 us fast sweep, 11.8 kHz) and its assembled frames ARE
the slow-axis 2D SLO (14.63 Hz measured). The pursuit dot (0.2 Hz Lissajous)
and the machine pupil tracker (~32.5 Hz) are the independent references.

Axis convention (test1 frame coords, see gaze_anim_raster.py / data.frame_truth):
  axis 0 (1000 px)  = FAST axis  -> VERTICAL gaze   (per-line trusted 1D shift)
  axis 1 (808 cols) = SLOW axis  -> HORIZONTAL gaze (per-line matched/aliased)

PER-LINE MEASUREMENT ENGINE (shared by M2/M3/M4/M5): every column of frame f
is compared against frame f-1 chain-aligned into frame-f coordinates:
  - lam_v : trusted vertical residual (1D NCC, subpixel)               [kHz]
  - prof  : horizontal NCC profile over offsets d in [-PADH, +PADH]    [kHz]
  - chain : per-frame absolute 2D anchor (incremental phase correlation,
            == data.frame_truth) — the 20 Hz-class absolute channel.
Total per-line position = chain(f) + residual; the methods differ in HOW they
turn the (aliased, multimodal) horizontal profile + anchors into a trajectory.

Sign conventions (verified in build): content displaced by +s between frames
means chain increment = -s, and pos_line = chain(f) + argpeak for BOTH axes
with the correlation definitions used below.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter

import data
import calib

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

WHICH = "test1"
CROPV = 160          # fast-axis rows cropped at top+bottom (warp borders)
PADH = 80            # horizontal profile half-width (px)
RDY_MAX = 60         # vertical residual search half-range (px)
MAXINC = 150         # max credible per-frame chain increment (px) — else blink/garbage
Q_FOV = 0.30         # per-line horizontal match-quality floor for "in FOV"
CONTRAST_FRAC = 0.5  # raw line contrast floor (x median)
DRIFT_HZ = 0.05      # drift-removal cutoff used uniformly by the evaluator
LINE_CACHE = os.path.join(CACHE, "khz2d_lines.npz")

TRK_DEG_X = data.UNITS.tracker_deg_per_unit_x
TRK_DEG_Y = data.UNITS.tracker_deg_per_unit_y


# ---------------------------------------------------------------------------
# small shared utilities
# ---------------------------------------------------------------------------


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 0 else np.nan


def fill_nan(x):
    x = np.asarray(x, float)
    i = np.arange(len(x)); m = np.isfinite(x)
    if not m.any():
        return np.zeros_like(x)
    return np.interp(i, i[m], x[m])


def _zcols(x):
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    return (x - mu) / (sd + 1e-9)


def _parab(y, k):
    """Sub-sample peak refinement around index k of profile y."""
    if 0 < k < len(y) - 1:
        a, b, c = y[k - 1], y[k], y[k + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            return k + 0.5 * (a - c) / den
    return float(k)


# ---------------------------------------------------------------------------
# 20 Hz-class anchor chain (the 2D SLO channel)
# ---------------------------------------------------------------------------


CHAIN_CACHE = os.path.join(CACHE, "khz2d_chain.npz")


def chain(rebuild: bool = False):
    """Per-frame absolute 2D anchor (cum. incremental registration), + validity.

    Registration = MEDIAN of per-strip (S=20) 2D matchTemplate shifts between
    consecutive frames — measured to be far more robust than full-frame phase
    correlation on this raster (the strips' median chain tracks the dot at
    |r|~0.85 vs ~0.62 for phase correlation, which mislocks on banding /
    low-overlap frames). Sign convention: cur[:, c] ~ prev[:, c + inc], i.e.
    pos = cumsum(inc) and pos_line = chain + rdx (same convention as the
    per-line engine residuals).

    Returns dict(t, x, y, inc_x, inc_y, ok, q, fps): x = horizontal (slow axis)
    px, y = vertical (fast axis) px, ok[f] False when the increment is not
    credible (blink / out-of-FOV / garbage match).
    """
    if os.path.exists(CHAIN_CACHE) and not rebuild:
        z = np.load(CHAIN_CACHE)
        return {k: z[k] for k in z.files}
    S, pad = 20, 80
    fps = None
    incs = [(0.0, 0.0)]
    qs = [1.0]
    prev = None
    cap = cv2.VideoCapture(data._slo_path(WHICH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    f = 0
    while True:
        okr, fr = cap.read()
        if not okr:
            break
        g = data._deband(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        cur = ((g - g.mean()) / (g.std() + 1e-9)).astype(np.float32)
        if prev is not None:
            H, W = cur.shape
            nstrip = W // S
            refp = cv2.copyMakeBorder(prev, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=0)
            fy = np.empty(nstrip); fx = np.empty(nstrip); fq = np.empty(nstrip)
            for s in range(nstrip):
                c = s * S
                r = cv2.matchTemplate(refp[:, c:c + S + 2 * pad], cur[:, c:c + S],
                                      cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                fy[s] = loc[1] - pad; fx[s] = loc[0] - pad; fq[s] = mx
            incs.append((float(np.median(fy)), float(np.median(fx))))
            qs.append(float(np.median(fq)))
        prev = cur
        f += 1
        if f % 200 == 0:
            print(f"  [chain] frame {f}")
    cap.release()
    incs = np.array(incs, np.float64)
    q = np.array(qs)
    ok = (q > 0.30) & (np.abs(incs).max(1) <= MAXINC)
    pos = np.cumsum(np.where(ok[:, None], incs, 0.0), 0)
    out = dict(t=np.arange(len(pos)) / fps, x=pos[:, 1], y=pos[:, 0],
               inc_x=np.where(ok, incs[:, 1], 0.0),
               inc_y=np.where(ok, incs[:, 0], 0.0),
               ok=ok, q=q, fps=np.float64(fps))
    np.savez(CHAIN_CACHE, **out)
    return out


# ---------------------------------------------------------------------------
# Per-line measurement engine (cached)
# ---------------------------------------------------------------------------


def _read_frames():
    """Yield (idx, raw_gray_float32) for every frame of the test1 SLO mp4."""
    cap = cv2.VideoCapture(data._slo_path(WHICH))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        i += 1
    cap.release()


def line_measurements(rebuild: bool = False, max_frames: int | None = None):
    """Per-line (per fast sweep) measurements vs the chain-aligned previous frame.

    Returns dict of arrays over N = (n_frames-1) * 808 lines:
      t       (N,)   line time, SLO clock (s)
      frame   (N,)   source frame index (>=1)
      col     (N,)   source column 0..807
      lam_v   (N,)   trusted vertical residual (px, subpixel; pos_y = chain_y + lam_v)
      rdx     (N,)   subpixel horizontal argpeak (px; pos_x = chain_x + rdx)
      prof    (N, 2*PADH+1) f16 horizontal NCC profile (offset d -> NCC)
      qv, qh  (N,)   vertical / horizontal peak NCC quality
      con     (N,)   raw line contrast (std)
      ok      (N,)   line credible (frame increment credible & finite)
    plus chain_* per-frame arrays and scalars fps, line_rate.
    """
    if os.path.exists(LINE_CACHE) and not rebuild and max_frames is None:
        z = np.load(LINE_CACHE)
        return {k: z[k] for k in z.files}

    ch = chain()
    fps = float(ch["fps"])
    W_off = 2 * PADH + 1
    T, FR, COL, LAMV, RDX, QV, QH, CON, OK = [], [], [], [], [], [], [], [], []
    PROF = []
    prev_raw = None
    nf = 0
    for f, raw in _read_frames():
        if max_frames is not None and f >= max_frames:
            break
        nf = f + 1
        if prev_raw is None:
            prev_raw = raw
            continue
        H, W = raw.shape
        ok_f = bool(ch["ok"][f])
        # chain-align prev into cur coords: content moved by s = -inc
        sx = -float(ch["inc_x"][f]); sy = -float(ch["inc_y"][f])
        if ok_f:
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            prevw = cv2.warpAffine(prev_raw, M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        else:
            prevw = prev_raw
        cur_db = data._deband(raw)[CROPV:H - CROPV]
        prv_db = data._deband(prevw)[CROPV:H - CROPV]
        Hc = cur_db.shape[0]
        curz = _zcols(cur_db); prvz = _zcols(prv_db)

        # --- vertical residual per column: c(l) = sum_n curz[n] * prvz[n+l] ---
        npad = Hc + 2 * RDY_MAX
        A = np.fft.rfft(curz, n=npad, axis=0)
        B = np.fft.rfft(prvz, n=npad, axis=0)
        cc = np.fft.irfft(np.conj(A) * B, n=npad, axis=0)  # [l] = sum cur[n]*prv[n+l]
        ccb = np.concatenate([cc[-RDY_MAX:], cc[:RDY_MAX + 1]], 0)  # l in [-R..R]
        pk = np.argmax(ccb, 0)
        lam = np.empty(W, np.float32); qv = np.empty(W, np.float32)
        for c in range(W):
            k = int(pk[c])
            lam[c] = _parab(ccb[:, c], k) - RDY_MAX
            qv[c] = ccb[k, c] / Hc
        # NOTE engine convention: cur[n] ~ prv[n - ry]  ->  peak at l = -ry,
        # and pos_y = chain_y + (-(-l)) ... verified: pos_y = chain_y + lam? No:
        # pos_y(line) = chain_y(f) - ry = chain_y(f) + lam.  (lam = -ry)

        # --- align cur columns vertically by lam, then horizontal profiles ---
        idx = np.arange(Hc, dtype=np.float32)[:, None] - lam[None, :]
        i0 = np.clip(np.floor(idx).astype(np.int32), 0, Hc - 2)
        frac = np.clip(idx - i0, 0.0, 1.0)
        cols = np.arange(W)[None, :].repeat(Hc, 0)
        cural = cur_db[i0, cols] * (1 - frac) + cur_db[i0 + 1, cols] * frac
        curalz = _zcols(cural)
        P = (prvz.T @ curalz) / Hc                       # (W, W): P[c+d, c] = NCC at offset d
        di = np.arange(-PADH, PADH + 1)
        src = np.arange(W)[None, :] + di[:, None]        # (W_off, W)
        bad = (src < 0) | (src >= W)
        src = np.clip(src, 0, W - 1)
        prof = P[src, np.arange(W)[None, :]]             # (W_off, W)
        prof[bad] = -1.0
        prof = prof.T                                    # (W, W_off)
        pk2 = np.argmax(prof, 1)
        rdx = np.empty(W, np.float32); qh = np.empty(W, np.float32)
        for c in range(W):
            k = int(pk2[c])
            rdx[c] = _parab(prof[c], k) - PADH
            qh[c] = prof[c, k]
        con = raw[CROPV:H - CROPV].std(0)

        tline = f / fps + (np.arange(W) + 0.5) / W / fps
        T.append(tline.astype(np.float64))
        FR.append(np.full(W, f, np.int32)); COL.append(np.arange(W, dtype=np.int16))
        LAMV.append(lam); RDX.append(rdx); QV.append(qv); QH.append(qh)
        CON.append(con.astype(np.float32))
        OK.append(np.full(W, ok_f))
        PROF.append(prof.astype(np.float16))
        prev_raw = raw
        if f % 100 == 0:
            print(f"  [lines] frame {f}  qh_med={np.median(qh):.2f} lam_med={np.median(np.abs(lam)):.1f}")

    out = dict(
        t=np.concatenate(T), frame=np.concatenate(FR), col=np.concatenate(COL),
        lam_v=np.concatenate(LAMV), rdx=np.concatenate(RDX),
        qv=np.concatenate(QV), qh=np.concatenate(QH), con=np.concatenate(CON),
        ok=np.concatenate(OK), prof=np.concatenate(PROF, 0),
        chain_t=ch["t"][:nf], chain_x=ch["x"][:nf], chain_y=ch["y"][:nf],
        chain_ok=ch["ok"][:nf],
        fps=np.float64(fps), line_rate=np.float64(fps * 808), padh=np.int64(PADH),
    )
    if max_frames is None:
        np.savez(LINE_CACHE, **out)
    return out


def fov_mask(lm):
    """In-FOV / credible-line mask shared by all per-line methods."""
    return (lm["ok"].astype(bool)
            & (lm["qh"] > Q_FOV)
            & (lm["con"] > CONTRAST_FRAC * np.median(lm["con"])))


# ---------------------------------------------------------------------------
# References (dot + tracker) and the shared clock offset
# ---------------------------------------------------------------------------

_REFS = None


def refs():
    """Stimulus dot and machine tracker in arcmin, plus the shared SLO->stimulus
    clock offset OFF.

    OFF is estimated ONCE at harness level (method-independent, then frozen for
    every method) from the raw per-line engine series (chain + rdx, fov-gated)
    against the dot, on a fine grid: the 0.2 Hz pursuit makes r very sensitive
    to OFF (~0.1 r per 100 ms), so a sloppy OFF would suppress every method
    equally but noisily. Falls back to the frame chain if the line cache is
    absent. OFF absorbs the hardware clock offset (~+2.2-2.6 s, sync OFF)
    plus mean pursuit latency.
    """
    global _REFS
    if _REFS is not None:
        return _REFS
    st = data.load_stimulus("pursuit")
    dot_t = st.t_s.astype(float)
    dot_x = st.x_deg * 60.0; dot_y = st.y_deg * 60.0
    tr = data.load_tracker(WHICH)
    trk_t = (tr.t_s - tr.t_s[0]).astype(float)
    trk_x = fill_nan(tr.right_x) * TRK_DEG_X * 60.0
    trk_y = fill_nan(tr.right_y) * TRK_DEG_Y * 60.0

    if os.path.exists(LINE_CACHE):
        z = np.load(LINE_CACHE)
        lm = {k: z[k] for k in ("t", "frame", "rdx", "qh", "con", "ok",
                                "chain_x", "line_rate")}
        pos = lm["chain_x"][lm["frame"]] + lm["rdx"]
        fs = float(lm["line_rate"])
        m = (lm["ok"].astype(bool) & (lm["qh"] > Q_FOV)
             & (lm["con"] > CONTRAST_FRAC * np.median(lm["con"])))
        sig = gaussian_filter1d(median_filter(fill_nan(np.where(m, pos, np.nan)), 7),
                                fs * 0.01)
        sig = _drift_removed(np.where(m, sig, np.nan), fs)
        tt = lm["t"]; mm = m
    else:
        ch = chain()
        sig = _drift_removed(np.where(ch["ok"], ch["x"], np.nan), ch["fps"])
        tt = ch["t"]; mm = np.isfinite(sig)
    best = (0.0, 0.0)
    for o in np.arange(-1.0, 6.0, 0.025):
        c = corr(sig[mm], np.interp(tt[mm], dot_t + o, dot_x))
        if np.isfinite(c) and abs(c) > abs(best[1]):
            best = (float(o), float(c))
    off = best[0]
    _REFS = dict(dot_t=dot_t, dot_x=dot_x, dot_y=dot_y,
                 trk_t=trk_t, trk_x=trk_x, trk_y=trk_y, off=off, off_r=best[1])
    return _REFS


def _drift_removed(v, fs):
    """Uniform slow-drift removal (cutoff DRIFT_HZ); keeps the 0.2 Hz pursuit band.
    Applied identically to every method (incremental chains drift). For high-rate
    series the low-pass is computed on a ~30 Hz decimated copy (identical result,
    avoids a 100k-tap kernel at line rate)."""
    v = fill_nan(v)
    if fs > 60.0:
        dec = max(1, int(round(fs / 30.0)))
        n = len(v) // dec * dec
        vd = v[:n].reshape(-1, dec).mean(1)
        fsd = fs / dec
        slow_d = gaussian_filter1d(vd, fsd / (2 * np.pi * DRIFT_HZ), mode="nearest")
        td = (np.arange(len(vd)) + 0.5) * dec
        slow = np.interp(np.arange(len(v)), td, slow_d)
    else:
        slow = gaussian_filter1d(v, fs / (2 * np.pi * DRIFT_HZ), mode="nearest")
    return v - slow


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def evaluate(t, x_px, y_px, valid, rate, label, smooth_ms: float = 0.0):
    """Score a reconstruction (px, chain coords) against dot + tracker.

    Uniform protocol for every method: optional light smoothing, slow-drift
    removal (DRIFT_HZ), per-axis affine calibration to the dot on valid samples
    (standard gaze calibration), then r + RMS arcmin vs dot and r vs tracker.
    Also reports a precision proxy: RMS of the >25 ms-detail component.
    """
    R = refs()
    t = np.asarray(t, float)
    vx = np.where(valid, np.asarray(x_px, float), np.nan)
    vy = np.where(valid, np.asarray(y_px, float), np.nan)
    if smooth_ms > 0 and rate > 0:
        k = max(1.0, rate * smooth_ms / 1000.0)
        vx = gaussian_filter1d(fill_nan(vx), k); vx[~valid] = np.nan
        vy = gaussian_filter1d(fill_nan(vy), k); vy[~valid] = np.nan
    hx = _drift_removed(vx, rate); hy = _drift_removed(vy, rate)
    dot_x = np.interp(t, R["dot_t"] + R["off"], R["dot_x"])
    dot_y = np.interp(t, R["dot_t"] + R["off"], R["dot_y"])
    trk_x = np.interp(t, R["trk_t"] + R["off"], R["trk_x"])
    trk_y = np.interp(t, R["trk_t"] + R["off"], R["trk_y"])

    out = dict(label=label, rate=float(rate), valid_frac=float(np.mean(valid)))
    for ax, h, d, tk in (("x", hx, dot_x, trk_x), ("y", hy, dot_y, trk_y)):
        m = valid & np.isfinite(h)
        r_dot = corr(h[m], d[m]); r_trk = corr(h[m], tk[m])
        if m.sum() > 20 and np.nanstd(h[m]) > 0:
            A = np.c_[h[m], np.ones(m.sum())]
            coef, *_ = np.linalg.lstsq(A, d[m], rcond=None)
            cal = coef[0] * h + coef[1]
            rms = float(np.sqrt(np.nanmean((cal[m] - d[m]) ** 2)))
            gain = float(coef[0])
        else:
            cal = h * np.nan; rms = np.nan; gain = np.nan
        if rate >= 200 and m.sum() > 200:
            k = rate * 0.025
            hf = cal - gaussian_filter1d(fill_nan(cal), k)
            prec = float(np.sqrt(np.nanmean(hf[m] ** 2)))
        else:
            prec = np.nan
        out[f"r_dot_{ax}"] = abs(r_dot) if np.isfinite(r_dot) else np.nan
        out[f"r_trk_{ax}"] = abs(r_trk) if np.isfinite(r_trk) else np.nan
        out[f"rms_{ax}"] = rms
        out[f"prec_{ax}"] = prec
        out[f"gain_{ax}"] = gain
        out[f"cal_{ax}"] = cal
        out[f"dot_{ax}"] = d
    out["t"] = t
    out["valid"] = valid
    return out


def summarize(ev):
    return (f"{ev['label']:<28s} {ev['rate']:>8.0f} Hz  "
            f"rdot x={ev['r_dot_x']:.3f} y={ev['r_dot_y']:.3f}  "
            f"rtrk x={ev['r_trk_x']:.3f} y={ev['r_trk_y']:.3f}  "
            f"RMS x={ev['rms_x']:.1f}' y={ev['rms_y']:.1f}'  "
            f"prec x={ev['prec_x']:.2f}' y={ev['prec_y']:.2f}'  "
            f"valid={ev['valid_frac']*100:.0f}%")


# ---------------------------------------------------------------------------
# method-output store
# ---------------------------------------------------------------------------


def save_method(name, t, x_px, y_px, valid, rate, extra=None):
    d = dict(t=t, x_px=x_px, y_px=y_px, valid=valid, rate=np.float64(rate))
    if extra:
        d.update(extra)
    np.savez(os.path.join(CACHE, f"khz2d_{name}.npz"), **d)
    return d


def load_method(name):
    p = os.path.join(CACHE, f"khz2d_{name}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build the per-line cache")
    ap.add_argument("--probe", type=int, default=0, help="probe N frames (no cache)")
    a = ap.parse_args()
    if a.probe:
        lm = line_measurements(rebuild=True, max_frames=a.probe)
        f = fov_mask(lm)
        print(f"lines={len(lm['t'])} qh med={np.median(lm['qh']):.3f} "
              f"qv med={np.median(lm['qv']):.3f} fov={f.mean()*100:.0f}% "
              f"rdx med|.|={np.median(np.abs(lm['rdx'])):.1f}px lam med|.|={np.median(np.abs(lm['lam_v'])):.1f}px")
    if a.build:
        lm = line_measurements(rebuild=True)
        print(f"built {LINE_CACHE}: {len(lm['t'])} lines @ {float(lm['line_rate']):.0f} Hz")
        R = refs()
        print(f"OFF={R['off']:.2f}s (chain-vs-dot r={R['off_r']:+.2f})")
