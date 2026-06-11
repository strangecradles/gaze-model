"""data.py — single data-loading module for the 2D gaze tracker (GOALS.md G1).

Loads the six imported assets the model builds on and exposes each with a
documented shape, dtype, and physical units. NO modeling / learned components
live here: the derived series (along-shift, coarse-anchor perp, frame-rate
registration truth) are fixed signal-processing reductions of the raw captures,
the inputs PLAN.md treats as trusted observations — not trained models.

Asset map (see progress.txt for the full reconnaissance):
  (a) atlas / enrollment map     -> normal/ raster frames  -> load_atlas()
  (b) raw ~12 kHz line-scan      -> calibration x-scan mp4  -> load_line_scan()
  (c) metric along-shift series  -> within-line registration -> along_shift()
  (d) coarse-anchor perp series  -> coarse-band atlas match   -> coarse_perp()
  (e) frame-rate registration    -> raster incr. registration -> frame_truth()
  (f) machine pupil tracker       -> calibration *.csv         -> load_tracker()

Coordinate convention everywhere: PERP = atlas row axis (slow axis, aliased,
~126 rows/deg); ALONG = atlas column axis (fast/scan axis, trusted via shift).

Run `python -m data --summary` for a table of every asset's shape, rate, units.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

# ---------------------------------------------------------------------------
# Units / measured constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Units:
    """Physical units and measured scale constants for every asset.

    Sample rates are in Hz. ``atlas_rows_per_deg`` / ``arcmin_per_row`` are the
    measured perp-axis scale (G2 re-measures and overwrites these as the source
    of truth; the values here are the sibling-project priors used for asserts
    and as fall-backs before calibration runs).
    """

    # spatial scale (perp / atlas row axis)
    atlas_rows_per_deg: float = 126.0          # alias spacing ~126 rows ~= 1 deg
    arcmin_per_row: float = 60.0 / 126.0       # ~0.476' per atlas row (perp)
    alias_spacing_rows: float = 126.0
    # along axis (atlas columns) — sub-pixel metric registration, trusted
    atlas_cols_along: int = 1200
    atlas_rows_perp: int = 600
    # capture sample rates
    line_rate_hz: float = 12000.0              # nominal; per-capture measured on load
    raster_fps: float = 14.6                   # frame-rate registration truth
    tracker_hz: float = 32.5                   # machine pupil tracker
    stimulus_fps: float = 60.0
    # kinematics / gate physics (perp rows/s)
    saccade_vmax_rows_per_s: float = 103000.0  # ~700 deg/s peak
    gate_threshold_hz: float = 820.0           # Vmax / alias_spacing
    # display geometry (stimuli_meta + user-provided test geometry)
    eye_to_display_mm: float = 147.9
    pixel_pitch_mm: float = 5.47e-3
    tracker_deg_per_unit_x: float = 0.611
    tracker_deg_per_unit_y: float = 0.461

    def rows_to_arcmin(self, rows: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(rows) * self.arcmin_per_row

    def arcmin_to_rows(self, arcmin: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(arcmin) / self.arcmin_per_row


UNITS = Units()

# ---------------------------------------------------------------------------
# Capture registry (newer calibration machine; cvr auto-classifier threshold N/A)
# ---------------------------------------------------------------------------

CAPTURES = {
    "Athton1": dict(stem="video_playback_Athton1_20260605_095610",
                    type="xscan", stimulus="calib_grid"),
    "test1":   dict(stem="video_playback_test1_20260605_100051",
                    type="raster", stimulus="pursuit"),
    "test2":   dict(stem="video_playback_test2_20260605_100257",
                    type="xscan", stimulus="pursuit"),
}


def _slo_path(which: str) -> str:
    return os.path.join(HERE, "calibration", CAPTURES[which]["stem"] + "_SLO_0.mp4")


def _csv_path(which: str) -> str:
    return os.path.join(HERE, "calibration", CAPTURES[which]["stem"] + ".csv")


# ---------------------------------------------------------------------------
# (a) Atlas / enrollment map
# ---------------------------------------------------------------------------


@dataclass
class AtlasData:
    frames: np.ndarray   # (n_frames, 600, 1200) float32 — raw raster frames, normalized
    phase: np.ndarray    # (n_frames, 1200) float32 — resonant phase per along column
    ref_map: np.ndarray  # (600, 1200) float32 — iteratively registered+averaged atlas
    offsets: np.ndarray  # (n_frames, 2) float32 — per-frame (dy=perp, dx=along) px
    rows_perp: int       # 600
    cols_along: int      # 1200

    def at(self, perp: float, along: float) -> float:
        """Bilinear sample of the reference atlas at (perp_row, along_col)."""
        return float(_bilinear(self.ref_map, perp, along))


def _norm_frame(t: np.ndarray) -> np.ndarray:
    t = np.log1p(t.astype(np.float32) - float(t.min()))
    s = t.std()
    return (t - t.mean()) / (s if s > 0 else 1.0)


def _phase_corr2d(ref: np.ndarray, mov: np.ndarray) -> tuple[int, int]:
    hy = np.hanning(ref.shape[0])[:, None]
    hx = np.hanning(ref.shape[1])[None, :]
    A = np.fft.rfft2(ref * hy * hx)
    B = np.fft.rfft2(mov * hy * hx)
    R = A * np.conj(B)
    m = np.abs(R)
    m[m == 0] = 1e-9
    cc = np.fft.fftshift(np.fft.irfft2(R / m, ref.shape))
    py, px = np.unravel_index(int(np.argmax(cc)), cc.shape)
    return py - ref.shape[0] // 2, px - ref.shape[1] // 2


def _shift_im(im: np.ndarray, dy: int, dx: int) -> np.ndarray:
    import cv2
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def load_atlas(person: str = "normal", n_frames: Optional[int] = None,
               rebuild: bool = False) -> AtlasData:
    """(a) Per-person atlas. frames (N,600,1200) f32, phase (N,1200), ref_map (600,1200).

    Units: rows = PERP (atlas-row, ~126 rows/deg, aliased); cols = ALONG (scan axis).
    Built by 3-pass iterative translation registration + averaging (subject fixated).
    """
    import tifffile

    cache = os.path.join(CACHE, f"atlas_{person}{'' if n_frames is None else f'_{n_frames}'}.npz")
    tiffs = sorted(glob.glob(os.path.join(HERE, person, "SLO_*.tiff")))
    if n_frames is not None:
        tiffs = tiffs[:n_frames]
    if not tiffs:
        raise FileNotFoundError(f"no SLO_*.tiff under {person}/")

    if os.path.exists(cache) and not rebuild:
        d = np.load(cache)
        a = AtlasData(d["frames"], d["phase"], d["ref_map"], d["offsets"],
                      int(d["rows_perp"]), int(d["cols_along"]))
    else:
        frames, phases = [], []
        for tp in tiffs:
            frames.append(_norm_frame(tifffile.imread(tp)))
            ph = os.path.splitext(tp)[0] + "_phase.csv"
            phases.append(np.loadtxt(ph).astype(np.float32))
        F = np.stack(frames, 0)
        P = np.stack(phases, 0)
        H, W = F[0].shape
        ref = F[0].copy()
        offs = np.zeros((len(F), 2), np.float32)
        for _ in range(3):
            acc = np.zeros((H, W)); cnt = np.zeros((H, W))
            for i in range(len(F)):
                dy, dx = _phase_corr2d(ref, F[i]); offs[i] = (dy, dx)
                s = _shift_im(F[i], dy, dx); v = ~np.isnan(s)
                acc[v] += s[v]; cnt[v] += 1
            mp = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
            sd = np.nanstd(mp)
            ref = (mp - np.nanmean(mp)) / (sd if sd > 0 else 1.0)
        a = AtlasData(F.astype(np.float32), P, ref.astype(np.float32), offs, H, W)
        np.savez(cache, frames=a.frames, phase=a.phase, ref_map=a.ref_map,
                 offsets=a.offsets, rows_perp=a.rows_perp, cols_along=a.cols_along)

    # --- load-time invariants ---
    assert a.frames.ndim == 3 and a.frames.shape[1:] == (a.rows_perp, a.cols_along), \
        f"atlas frame shape {a.frames.shape}"
    assert a.ref_map.shape == (a.rows_perp, a.cols_along)
    assert a.phase.shape == (a.frames.shape[0], a.cols_along)
    assert np.isfinite(a.ref_map).all(), "ref_map has non-finite values"
    assert a.rows_perp == UNITS.atlas_rows_perp and a.cols_along == UNITS.atlas_cols_along
    return a


# ---------------------------------------------------------------------------
# (b) Raw ~12 kHz line-scan stream
# ---------------------------------------------------------------------------


@dataclass
class LineScan:
    sweeps: np.ndarray   # (n_sweeps, along_len) float32 — 1D along-line profiles
    line_rate_hz: float  # measured sweeps/s (~12 kHz)
    along_len: int       # 1000
    n_total: int         # full sweep count in the capture (>= sweeps.shape[0])
    which: str
    capture_type: str    # 'xscan' | 'raster'


def line_scan_meta(which: str = "test2") -> dict:
    """Cheap metadata (no pixel load): n_total sweeps, along_len, measured line rate."""
    import cv2
    path = _slo_path(which)
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    dur = n / fps if fps else float("nan")
    # x-scan frame is (h=along, w=sweeps); n_sweeps = w * n_frames
    n_sweeps = w * n
    return dict(along_len=h, sweeps_per_frame=w, n_frames=n, fps=fps, dur_s=dur,
                n_total=n_sweeps, line_rate_hz=n_sweeps / dur if dur else float("nan"),
                path=path)


def load_line_scan(which: str = "test2", max_sweeps: Optional[int] = None) -> LineScan:
    """(b) Raw line-scan stream. sweeps (n, along_len) f32; row = one ~80 us sweep.

    Each mp4 frame is (along_len=1000, n_sweeps_per_frame=808); transposed and
    concatenated across frames into a continuous ~12 kHz stream. ``max_sweeps``
    caps how many are decoded into memory (full stream is multi-GB).
    Units: amplitude = reflectance (a.u.); axis 1 index = ALONG sample.
    """
    import cv2
    meta = line_scan_meta(which)
    path = meta["path"]
    cap = cv2.VideoCapture(path)
    chunks = []
    got = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        sw = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32).T  # (n_sweeps, along)
        if max_sweeps is not None and got + sw.shape[0] > max_sweeps:
            sw = sw[: max_sweeps - got]
        chunks.append(sw); got += sw.shape[0]
        if max_sweeps is not None and got >= max_sweeps:
            break
    cap.release()
    sweeps = np.concatenate(chunks, 0) if chunks else np.zeros((0, meta["along_len"]), np.float32)

    ls = LineScan(sweeps, float(meta["line_rate_hz"]), int(meta["along_len"]),
                  int(meta["n_total"]), which, CAPTURES[which]["type"])
    # invariants
    assert ls.sweeps.ndim == 2 and ls.sweeps.shape[1] == ls.along_len, \
        f"line-scan shape {ls.sweeps.shape}"
    assert np.isfinite(ls.sweeps).all(), "line-scan has non-finite samples"
    assert 8000.0 < ls.line_rate_hz < 16000.0, f"line rate {ls.line_rate_hz} Hz out of ~12kHz band"
    return ls


# ---------------------------------------------------------------------------
# Shared 1D / 2D helpers
# ---------------------------------------------------------------------------


def _ncc_shift(ref: np.ndarray, mov: np.ndarray, max_shift: int = 25) -> float:
    """Sub-pixel along-axis shift of ``mov`` relative to ``ref`` (parabolic refine)."""
    a = ref - ref.mean(); b = mov - mov.mean()
    n = len(a)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    c = np.correlate(b, a, "full") / denom
    lags = np.arange(-n + 1, n)
    sel = (lags >= -max_shift) & (lags <= max_shift)
    idx = np.flatnonzero(sel)
    k = idx[int(np.argmax(c[idx]))]
    off = 0.0
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return float(lags[k] + off)


def _bilinear(img: np.ndarray, y: float, x: float) -> float:
    H, W = img.shape
    y = min(max(y, 0.0), H - 1.0); x = min(max(x, 0.0), W - 1.0)
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
    fy, fx = y - y0, x - x0
    return float((img[y0, x0] * (1 - fy) * (1 - fx) + img[y0, x1] * (1 - fy) * fx +
                  img[y1, x0] * fy * (1 - fx) + img[y1, x1] * fy * fx))


# ---------------------------------------------------------------------------
# (c) Metric along-shift series (trusted along observation)
# ---------------------------------------------------------------------------


def along_shift(which: str = "test2", max_sweeps: int = 20000,
                rebuild: bool = False) -> np.ndarray:
    """(c) Trusted metric along-shift series, shape (n,) float32, units = ALONG px.

    Whole-line incremental NCC shift between consecutive sweeps, cumulatively
    summed (per-window registration is too noisy; the whole line is the reliable
    metric channel, r~0.60 vs saccade-following truth). Mean-removed (relative).
    """
    cache = os.path.join(CACHE, f"alongshift_{which}_{max_sweeps}.npy")
    if os.path.exists(cache) and not rebuild:
        return np.load(cache)
    ls = load_line_scan(which, max_sweeps=max_sweeps)
    sw = ls.sweeps
    inc = np.zeros(len(sw), np.float32)
    prev = sw[0]
    for i in range(1, len(sw)):
        inc[i] = _ncc_shift(prev, sw[i])
        prev = sw[i]
    pos = np.cumsum(inc)
    pos -= pos.mean()
    pos = pos.astype(np.float32)
    np.save(cache, pos)
    assert pos.ndim == 1 and len(pos) == len(sw) and np.isfinite(pos).all()
    return pos


# ---------------------------------------------------------------------------
# (d) Coarse-anchor perp series
# ---------------------------------------------------------------------------


def coarse_perp(which: str = "test2", atlas: Optional[AtlasData] = None,
                block: int = 32, max_sweeps: int = 20000, sigma: float = 8.0,
                rebuild: bool = False) -> np.ndarray:
    """(d) Coarse-anchor perpendicular series, shape (n_blocks,) float32, units = atlas rows.

    Block-averaged sweeps matched against the spatial-low-pass (coarse-band)
    atlas. Coarse band aliases differently than fine -> broad ~0.5 deg / ~43'
    absolute perp anchor for reseeding (NOT a precise/veto measurement).
    """
    import cv2
    from scipy.ndimage import gaussian_filter1d
    cache = os.path.join(CACHE, f"coarseperp_{which}_{block}_{max_sweeps}.npy")
    if os.path.exists(cache) and not rebuild:
        return np.load(cache)
    if atlas is None:
        atlas = load_atlas()
    R = atlas.ref_map.astype(np.float32)
    Rc = gaussian_filter1d(R, sigma, axis=1)  # coarse-band atlas (along-axis low-pass)

    ls = load_line_scan(which, max_sweeps=max_sweeps)
    sw = ls.sweeps
    nb = len(sw) // block
    blk = sw[: nb * block].reshape(nb, block, -1).mean(1)
    Wv = blk.shape[1]

    # one-time scale fit of x-scan along-length onto atlas along-width
    strip = blk[: min(nb, 50)].mean(0)
    strip = (strip - strip.mean()) / (strip.std() + 1e-6)
    best = (-2.0, 0.5)
    for s in np.linspace(0.38, 1.0, 32):
        L = max(8, int(Wv * s))
        tt = cv2.resize(strip.reshape(1, -1), (L, 1)).astype(np.float32)
        tt = (tt - tt.mean()) / (tt.std() + 1e-6)
        if L > Rc.shape[1]:
            continue
        r = cv2.matchTemplate(Rc, tt, cv2.TM_CCOEFF_NORMED)
        if r.max() > best[0]:
            best = (float(r.max()), float(s))
    scale = best[1]
    L = max(8, int(Wv * scale))

    out = np.zeros(nb, np.float32)
    for i in range(nb):
        tt = cv2.resize(blk[i].reshape(1, -1), (L, 1)).astype(np.float32).ravel()
        tc = gaussian_filter1d(tt, sigma)
        tc = (tc - tc.mean()) / (tc.std() + 1e-6)
        prof = cv2.matchTemplate(Rc, tc.reshape(1, -1), cv2.TM_CCOEFF_NORMED).max(1)
        p = int(prof.argmax())
        # parabolic sub-pixel
        if 0 < p < len(prof) - 1:
            a, b, c = prof[p - 1], prof[p], prof[p + 1]; den = a - 2 * b + c
            out[i] = p + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
        else:
            out[i] = float(p)
    np.save(cache, out)
    assert out.ndim == 1 and len(out) == nb and np.isfinite(out).all()
    return out


# ---------------------------------------------------------------------------
# (e) Frame-rate full-frame-registration truth
# ---------------------------------------------------------------------------


@dataclass
class FrameTruth:
    t_s: np.ndarray      # (n_frames,) seconds (SLO clock)
    perp_px: np.ndarray  # (n_frames,) perp displacement (atlas-row equivalent px)
    along_px: np.ndarray # (n_frames,) along displacement px
    fps: float
    which: str


def _deband(g: np.ndarray) -> np.ndarray:
    g = g - g.mean(1, keepdims=True)   # per-row mean
    g = g - g.mean(0, keepdims=True)   # per-col mean
    return g


def frame_truth(which: str = "test1", max_frames: Optional[int] = None,
                rebuild: bool = False) -> FrameTruth:
    """(e) Frame-rate full-frame-registration truth from the RASTER capture.

    De-band (per-row + per-col mean) then INCREMENTAL phase-correlation between
    consecutive frames (fixed-ref fails as the patch leaves FOV); cumulative
    sum gives per-frame 2D gaze (perp=dy, along=dx) in px. Validated in sibling
    work vs machine tracker (ex~-0.86 right_x, ey~-0.76 right_y).
    Units: px in the raster frame (perp=vertical/slow, along=horizontal/fast).
    """
    import cv2
    if CAPTURES[which]["type"] != "raster":
        raise ValueError(f"frame_truth requires a raster capture; {which} is {CAPTURES[which]['type']}")
    tag = "all" if max_frames is None else str(max_frames)
    cache = os.path.join(CACHE, f"frametruth_{which}_{tag}.npz")
    if os.path.exists(cache) and not rebuild:
        d = np.load(cache)
        return FrameTruth(d["t_s"], d["perp_px"], d["along_px"], float(d["fps"]), which)

    path = _slo_path(which)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    incr = [(0.0, 0.0)]
    prev = None
    nread = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = _deband(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
        if prev is not None:
            dy, dx = _phase_corr2d(prev, g)
            incr.append((float(dy), float(dx)))
        prev = g; nread += 1
        if max_frames is not None and nread >= max_frames:
            break
    cap.release()
    incr = np.array(incr, np.float32)
    pos = np.cumsum(incr, 0)
    perp = pos[:, 0] - pos[:, 0].mean()
    along = pos[:, 1] - pos[:, 1].mean()
    t = np.arange(len(perp)) / (fps if fps else 1.0)
    ft = FrameTruth(t.astype(np.float32), perp.astype(np.float32), along.astype(np.float32),
                    float(fps), which)
    np.savez(cache, t_s=ft.t_s, perp_px=ft.perp_px, along_px=ft.along_px, fps=ft.fps)
    assert ft.perp_px.shape == ft.along_px.shape == ft.t_s.shape
    assert np.isfinite(ft.perp_px).all() and np.isfinite(ft.along_px).all()
    return ft


# ---------------------------------------------------------------------------
# (f) Machine pupil tracker
# ---------------------------------------------------------------------------


@dataclass
class MachineTrack:
    t_s: np.ndarray        # (n,) seconds since CSV start
    right_x: np.ndarray    # (n,) eye-tracker units (~0.611 deg/unit horizontal)
    right_y: np.ndarray    # (n,) eye-tracker units (~0.461 deg/unit vertical)
    right_area_mm2: np.ndarray  # (n,) pupil area
    video_frame: np.ndarray     # (n,) stimulus dot frame index
    rate_hz: float
    which: str


def load_tracker(which: str = "test2") -> MachineTrack:
    """(f) Machine pupil tracker right_x/right_y, ~32.5 Hz. Units: eye-tracker
    degrees-of-arc scaled (~0.5 deg/unit). video_frame indexes the stimulus dot."""
    import pandas as pd
    df = pd.read_csv(_csv_path(which))
    t = pd.to_datetime(df["timestamp"])
    t_s = (t - t.iloc[0]).dt.total_seconds().to_numpy().astype(np.float64)
    dur = t_s[-1] - t_s[0] if len(t_s) > 1 else 1.0
    rate = len(df) / dur if dur > 0 else float("nan")
    mt = MachineTrack(
        t_s,
        df["right_x"].astype(np.float64).to_numpy(),
        df["right_y"].astype(np.float64).to_numpy(),
        df["right_area_mm2"].astype(np.float64).to_numpy(),
        df["video_frame"].astype(np.float64).to_numpy(),
        float(rate), which)
    assert mt.right_x.shape == mt.right_y.shape == mt.t_s.shape
    assert 20.0 < mt.rate_hz < 60.0, f"tracker rate {mt.rate_hz} Hz off ~32.5"
    return mt


# ---------------------------------------------------------------------------
# Stimulus (dot path) — used by calibration / validation
# ---------------------------------------------------------------------------


@dataclass
class Stimulus:
    t_s: np.ndarray
    x_px: np.ndarray
    y_px: np.ndarray
    x_deg: np.ndarray
    y_deg: np.ndarray
    phase: np.ndarray
    fps: float


def load_stimulus(name: str = "pursuit") -> Stimulus:
    """Stimulus dot path. Degrees via atan2((px-center)*pitch, D) from stimuli_meta geometry."""
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, "stimuli", f"{name}.csv"))
    cx, cy = 1920 / 2.0, 1080 / 2.0
    xdeg = np.degrees(np.arctan2((df["x_px"].to_numpy() - cx) * UNITS.pixel_pitch_mm,
                                 UNITS.eye_to_display_mm))
    ydeg = np.degrees(np.arctan2((df["y_px"].to_numpy() - cy) * UNITS.pixel_pitch_mm,
                                 UNITS.eye_to_display_mm))
    return Stimulus(df["t_sec"].to_numpy(), df["x_px"].to_numpy(), df["y_px"].to_numpy(),
                    xdeg, ydeg, df["phase"].to_numpy(), UNITS.stimulus_fps)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def summary() -> str:
    rows = []
    rows.append(("asset", "shape", "rate", "units"))
    # (a)
    a = load_atlas()
    rows.append(("(a) atlas ref_map", str(a.ref_map.shape), "-",
                 f"rows=perp(~{UNITS.atlas_rows_per_deg:.0f}/deg), cols=along"))
    rows.append(("(a) atlas frames", str(a.frames.shape), "-", "normalized reflectance"))
    rows.append(("(a) atlas phase", str(a.phase.shape), "-", "resonant phase/col"))
    # (b)
    m = line_scan_meta("test2")
    rows.append(("(b) line-scan test2", f"({m['n_total']}, {m['along_len']})",
                 f"{m['line_rate_hz']:.0f} Hz", "reflectance; axis1=along"))
    m1 = line_scan_meta("Athton1")
    rows.append(("(b) line-scan Athton1", f"({m1['n_total']}, {m1['along_len']})",
                 f"{m1['line_rate_hz']:.0f} Hz", "reflectance; axis1=along"))
    # (c)
    c = along_shift("test2", max_sweeps=4000)
    rows.append(("(c) along-shift test2", f"({c.shape[0]},)*", f"{m['line_rate_hz']:.0f} Hz",
                 "along px (trusted)"))
    # (d)
    d = coarse_perp("test2", atlas=a, max_sweeps=4000)
    rows.append(("(d) coarse-perp test2", f"({d.shape[0]},)*",
                 f"{m['line_rate_hz']/32:.0f} Hz (blk32)", "atlas rows (~43')"))
    # (e)
    ft = frame_truth("test1", max_frames=60)
    rows.append(("(e) frame-truth test1", f"({ft.perp_px.shape[0]},2)*", f"{ft.fps:.1f} fps",
                 "px (perp,along)"))
    # (f)
    t = load_tracker("test2")
    rows.append(("(f) machine tracker", f"({t.t_s.shape[0]},)", f"{t.rate_hz:.1f} Hz",
                 "right_x/y eye-tracker units"))

    w = [max(len(str(r[i])) for r in rows) for i in range(4)]
    lines = ["  (* = computed on a capped slice for the summary)"]
    for i, r in enumerate(rows):
        lines.append("  " + " | ".join(str(r[j]).ljust(w[j]) for j in range(4)))
        if i == 0:
            lines.append("  " + "-+-".join("-" * w[j] for j in range(4)))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="2D gaze tracker data assets")
    p.add_argument("--summary", action="store_true", help="print asset table")
    args = p.parse_args()
    if args.summary:
        print("2D Gaze Tracker — data assets")
        print(summary())
    else:
        p.print_help()
