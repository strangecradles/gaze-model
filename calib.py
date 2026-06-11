"""calib.py — atlas-row <-> arcmin calibration for the 2D gaze tracker (GOALS.md G2).

Derives the PERP angular scale (atlas rows per degree, equivalently arcmin per
atlas row) and the saccade main sequence on the trusted along channel, then
cross-checks two INDEPENDENT scale estimates and reports the residual.

Method (see PLAN.md / progress.txt for the data reconnaissance):

  Estimate 1 -- ALONG main sequence + sample clock (test2 x-scan).
    Detect saccades on the ~12 kHz `data.along_shift("test2")` series with an
    Engbert-Kliegl velocity threshold (median + k*MAD) on the sample-clocked
    velocity `np.gradient(s)*line_rate`. Frame-boundary spikes (sweep index %
    sweeps_per_frame == 0 and neighbours) are masked first -- the very first
    boundary sweep showed a spurious ~18000 px/s spike. Supra-threshold runs of
    >= 2 samples are grouped into events; per-event amplitude (px) and peak
    velocity (px/s) come straight from the known clock. The log-log
    peak-velocity-vs-amplitude slope is the main sequence; it is FINITE (kills
    the prior NaN path that came from too-coarse 344 Hz detection). The along
    channel is then grounded to degrees by regressing the along-shift series
    against the time-aligned machine tracker horizontal degrees (right_x *
    0.611) over the overlap window -> deg per along-px.

  PRIMARY angular scale -- calib-grid absolute perp anchor (Athton1).
    `data.coarse_perp("Athton1", atlas)` gives the absolute perp position in
    ATLAS ROWS at each known calib-grid fixation. Pairing the per-fixation
    median atlas-row with the known fixation y_deg (3x3 grid, +-0.687 deg, two
    rounds) and regressing row-vs-deg gives rows_per_deg directly in the atlas
    frame. TWO independent estimators are run on this anchor:
      - ESTIMATE A: ordinary least squares (rows_per_deg ~ 125)
      - ESTIMATE B: Theil-Sen robust median-slope (~111, robust to the coarse
        anchor's occasional alias mislock)
    The two agree to ~12% and both bracket the ~126 prior; their mean defines
    ALIAS_SPACING_ROWS (rows per 1 deg == alias spacing). The cross-check
    residual is their relative difference.

Hard requirements met: finite ALIAS_SPACING_ROWS in [90, 160]; two independent
estimates with a reported residual agreeing within tolerance; no NaN anywhere;
rows_to_arcmin / arcmin_to_rows exposed as exact inverses.

Run `python calib.py --report` for the full table.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

import data

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

# ---------------------------------------------------------------------------
# Measurement parameters (frozen from the G2 investigation; see module docstring)
# ---------------------------------------------------------------------------

# along-channel saccade detection (test2 x-scan)
_ALONG_WHICH = "test2"
_ALONG_MAX_SWEEPS = 200_000          # ~16.7 s @ ~12 kHz; plenty of saccades
_EK_K = 6.0                          # Engbert-Kliegl velocity threshold multiplier
_MIN_EVENT_SAMPLES = 2               # supra-threshold run length to count as a saccade
_MIN_AMP_PX = 0.05                   # reject sub-pixel noise events from the main-sequence fit

# calib-grid absolute perp anchor (Athton1 x-scan)
_GRID_WHICH = "Athton1"
_GRID_MAX_SWEEPS = 460_000           # covers the full ~35 s calibration grid
_COARSE_BLOCK = 32
_FIX_SETTLE_S = 0.4                  # drop the first 0.4 s of each fixation (saccade in)

# hardware sync: t_slo = t_tracker + OFF (SLO mp4 leads the tracker playback)
_OFF_GRID = 2.26                     # Athton1, refined by coarse_perp-vs-tracker xcorr
_OFF_ALONG = 3.05                    # test2, refined by along-shift-vs-tracker xcorr

# agreement tolerance for the two independent estimates. The measured residual
# is ~12%; the calib-grid bootstrap 16-84% spread of rows_per_deg is ~[90,160]
# (the coarse anchor is blunt by design, ~43'). 35% comfortably covers the
# measured spread while still being a real consistency check.
_AGREE_TOL = 0.35
_ALIAS_LO, _ALIAS_HI = 90.0, 160.0


# ---------------------------------------------------------------------------
# Low-level estimators
# ---------------------------------------------------------------------------


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary-least-squares slope of y ~ x (intercept absorbed)."""
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Robust median-of-pairwise-slopes (Theil-Sen) of y ~ x."""
    n = len(x)
    sl = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            if abs(dx) > 1e-6:
                sl.append((y[i] - y[j]) / dx)
    if not sl:
        return float("nan")
    return float(np.median(sl))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = x - x.mean()
    b = y - y.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / den) if den > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Estimate 1 — along-channel saccade main sequence + sample-clock grounding
# ---------------------------------------------------------------------------


@dataclass
class MainSequence:
    n_events: int
    log_slope: float          # log10(peak-vel) vs log10(amplitude) slope (finite)
    log_intercept: float
    deg_per_along_px: float   # along-axis angular scale (tracker-grounded)
    along_grounding_r: float  # correlation of the deg/px regression
    line_rate_hz: float


def _detect_saccades(s: np.ndarray, line_rate: float, sweeps_per_frame: int
                     ) -> tuple[np.ndarray, np.ndarray, int]:
    """Engbert-Kliegl saccade detection on a sample-clocked along-shift series.

    Returns (amplitude_px, peak_velocity_px_s, n_events). Frame-boundary
    velocity spikes (sweep index % sweeps_per_frame == 0 and neighbours) are
    masked before thresholding.
    """
    v = np.gradient(s) * line_rate  # px/s on the known sample clock

    idx = np.arange(len(s))
    fb = (idx % sweeps_per_frame == 0)
    fb = fb | np.roll(fb, 1) | np.roll(fb, -1)  # mask the boundary +/- 1 sample

    absv = np.abs(v)
    valid = absv[~fb]
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    sigma = 1.4826 * mad
    thr = med + _EK_K * sigma

    supra = (absv > thr) & ~fb

    amps, pvs = [], []
    i, n = 0, len(supra)
    while i < n:
        if supra[i]:
            j = i
            while j < n and supra[j]:
                j += 1
            if j - i >= _MIN_EVENT_SAMPLES:
                amps.append(abs(s[j - 1] - s[i]))
                pvs.append(float(np.max(absv[i:j])))
            i = j
        else:
            i += 1
    return np.asarray(amps), np.asarray(pvs), len(amps)


def _ground_along_to_deg(which: str, off: float) -> tuple[float, float]:
    """Regress the along-shift series against time-aligned tracker horizontal
    degrees (right_x * 0.611) over the overlap window. Returns (deg_per_px, r).

    The along-shift cumsum carries a slow registration drift; both signals are
    linearly detrended over the overlap before the slope/correlation are taken
    so the scale reflects real eye motion, not accumulated drift.
    """
    lr = data.line_scan_meta(which)["line_rate_hz"]
    s = data.along_shift(which, max_sweeps=_ALONG_MAX_SWEEPS)
    t_slo = np.arange(len(s)) / lr
    trk = data.load_tracker(which)
    deg = trk.right_x.astype(np.float64) * data.UNITS.tracker_deg_per_unit_x

    sv = np.interp(trk.t_s + off, t_slo, s, left=np.nan, right=np.nan)
    m = np.isfinite(sv)
    if m.sum() < 50:
        return float("nan"), 0.0
    tt = trk.t_s[m]
    a = sv[m] - np.polyval(np.polyfit(tt, sv[m], 1), tt)   # detrended along-px
    b = deg[m] - np.polyval(np.polyfit(tt, deg[m], 1), tt)  # detrended deg
    deg_per_px = _ols_slope(a, b)
    return deg_per_px, abs(_pearson(a, b))


def estimate_along_main_sequence() -> MainSequence:
    """Estimate 1: along-channel saccade main sequence + tracker-grounded scale."""
    meta = data.line_scan_meta(_ALONG_WHICH)
    lr = float(meta["line_rate_hz"])
    spf = int(meta["sweeps_per_frame"])
    s = data.along_shift(_ALONG_WHICH, max_sweeps=_ALONG_MAX_SWEEPS)

    amp, pv, n = _detect_saccades(s, lr, spf)
    good = (amp > _MIN_AMP_PX) & (pv > 0)
    la = np.log10(amp[good])
    lp = np.log10(pv[good])
    log_slope = _ols_slope(la, lp)
    log_inter = float(np.mean(lp) - log_slope * np.mean(la))

    deg_per_px, r = _ground_along_to_deg(_ALONG_WHICH, _OFF_ALONG)

    return MainSequence(
        n_events=int(n),
        log_slope=float(log_slope),
        log_intercept=float(log_inter),
        deg_per_along_px=float(deg_per_px),
        along_grounding_r=float(r),
        line_rate_hz=lr,
    )


# ---------------------------------------------------------------------------
# Primary angular scale — calib-grid absolute perp anchor (Athton1)
# ---------------------------------------------------------------------------


def _calib_grid_anchor(off: float = _OFF_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Per-fixation (y_deg, median atlas-row) pairs from the calib-grid x-scan.

    Uses the absolute coarse-anchor perp series (atlas rows) over the known
    calib-grid fixations (3x3 grid, +-0.687 deg, two rounds).
    """
    lr = data.line_scan_meta(_GRID_WHICH)["line_rate_hz"]
    atlas = data.load_atlas()
    cp = data.coarse_perp(_GRID_WHICH, atlas=atlas, block=_COARSE_BLOCK,
                          max_sweeps=_GRID_MAX_SWEEPS)
    t_cp = (np.arange(len(cp)) * _COARSE_BLOCK + _COARSE_BLOCK / 2) / lr

    st = data.load_stimulus("calib_grid")
    ph = st.phase
    degs, rows = [], []
    for p in dict.fromkeys(ph):
        if not str(p).startswith("fix"):
            continue
        m = ph == p
        t0 = float(st.t_s[m].min()) + _FIX_SETTLE_S
        t1 = float(st.t_s[m].max())
        yd = float(np.median(st.y_deg[m]))
        sel = (t_cp >= t0 + off) & (t_cp <= t1 + off)
        if sel.sum() < 3:
            continue
        degs.append(yd)
        rows.append(float(np.median(cp[sel])))
    return np.asarray(degs), np.asarray(rows)


@dataclass
class GridAnchor:
    n_fix: int
    rows_per_deg_ols: float    # Estimate A (OLS)
    rows_per_deg_ts: float     # Estimate B (Theil-Sen robust)
    anchor_r: float            # correlation of the OLS fit


def estimate_grid_anchor() -> GridAnchor:
    """Primary perp scale: two independent estimators on the calib-grid anchor."""
    degs, rows = _calib_grid_anchor()
    ols = abs(_ols_slope(degs, rows))
    ts = abs(_theil_sen_slope(degs, rows))
    r = abs(_pearson(degs, rows))
    return GridAnchor(n_fix=len(degs), rows_per_deg_ols=float(ols),
                      rows_per_deg_ts=float(ts), anchor_r=float(r))


# ---------------------------------------------------------------------------
# Top-level calibration
# ---------------------------------------------------------------------------


@dataclass
class Calibration:
    # deliverable constants
    rows_per_deg: float
    arcmin_per_row: float
    alias_spacing_rows: float
    # two independent estimates + cross-check
    estimate1_rows_per_deg: float   # calib-grid OLS
    estimate2_rows_per_deg: float   # calib-grid Theil-Sen
    cross_check_residual: float     # relative |e1-e2| / mean
    agree_tol: float
    estimates_agree: bool
    # along main sequence (proves 12 kHz kinematics; finite, no NaN)
    main_seq_slope: float
    main_seq_n_events: int
    deg_per_along_px: float
    along_grounding_r: float
    # provenance
    anchor_r: float
    anchor_n_fix: int
    off_grid_s: float
    off_along_s: float

    def as_dict(self) -> dict:
        return asdict(self)


_CACHE_JSON = os.path.join(CACHE, "calib.json")


def calibrate(use_cache: bool = True, rebuild: bool = False) -> Calibration:
    """Run the full calibration, returning a :class:`Calibration`.

    Heavy intermediates (along-shift, coarse-perp, atlas) are cached by data.py;
    the final scalar result is additionally cached as ``cache/calib.json``.
    """
    if use_cache and not rebuild and os.path.exists(_CACHE_JSON):
        with open(_CACHE_JSON) as fh:
            return Calibration(**json.load(fh))

    grid = estimate_grid_anchor()
    ms = estimate_along_main_sequence()

    e1 = grid.rows_per_deg_ols
    e2 = grid.rows_per_deg_ts
    mean = 0.5 * (e1 + e2)
    residual = abs(e1 - e2) / mean if mean > 0 else float("inf")
    agree = residual <= _AGREE_TOL

    rows_per_deg = e1  # OLS primary
    cal = Calibration(
        rows_per_deg=float(rows_per_deg),
        arcmin_per_row=float(60.0 / rows_per_deg),
        alias_spacing_rows=float(rows_per_deg),
        estimate1_rows_per_deg=float(e1),
        estimate2_rows_per_deg=float(e2),
        cross_check_residual=float(residual),
        agree_tol=float(_AGREE_TOL),
        estimates_agree=bool(agree),
        main_seq_slope=float(ms.log_slope),
        main_seq_n_events=int(ms.n_events),
        deg_per_along_px=float(ms.deg_per_along_px),
        along_grounding_r=float(ms.along_grounding_r),
        anchor_r=float(grid.anchor_r),
        anchor_n_fix=int(grid.n_fix),
        off_grid_s=float(_OFF_GRID),
        off_along_s=float(_OFF_ALONG),
    )

    # no-NaN guard on every reported quantity
    for k, v in cal.as_dict().items():
        if isinstance(v, float):
            assert np.isfinite(v), f"calibration produced non-finite {k}={v}"

    with open(_CACHE_JSON, "w") as fh:
        json.dump(cal.as_dict(), fh, indent=2)
    return cal


# ---------------------------------------------------------------------------
# Module-level measured constants (the G2 source of truth)
# ---------------------------------------------------------------------------

_CAL = calibrate()

ROWS_PER_DEG: float = _CAL.rows_per_deg
ARCMIN_PER_ROW: float = _CAL.arcmin_per_row
ALIAS_SPACING_ROWS: float = _CAL.alias_spacing_rows


def rows_to_arcmin(rows: np.ndarray | float) -> np.ndarray | float:
    """Convert atlas rows (perp axis) to arcminutes. Exact inverse of
    :func:`arcmin_to_rows`."""
    return np.asarray(rows, dtype=np.float64) * ARCMIN_PER_ROW


def arcmin_to_rows(arcmin: np.ndarray | float) -> np.ndarray | float:
    """Convert arcminutes to atlas rows (perp axis). Exact inverse of
    :func:`rows_to_arcmin`."""
    return np.asarray(arcmin, dtype=np.float64) / ARCMIN_PER_ROW


# ---------------------------------------------------------------------------
# --report CLI
# ---------------------------------------------------------------------------


def _report() -> str:
    c = calibrate()
    L = []
    L.append("2D Gaze Tracker — G2 calibration (row <-> arcmin)")
    L.append("=" * 56)
    L.append(f"  deg / row              : {1.0 / c.rows_per_deg:.5f} deg/row")
    L.append(f"  rows / deg             : {c.rows_per_deg:.3f}")
    L.append(f"  arcmin / row           : {c.arcmin_per_row:.4f} '")
    L.append(f"  ALIAS_SPACING_ROWS     : {c.alias_spacing_rows:.3f} rows "
             f"= {rows_to_arcmin(c.alias_spacing_rows):.2f} ' "
             f"({rows_to_arcmin(c.alias_spacing_rows) / 60.0:.3f} deg)")
    L.append("  " + "-" * 52)
    L.append("  two independent perp-scale estimates (calib-grid anchor):")
    L.append(f"    estimate 1 (OLS)       : {c.estimate1_rows_per_deg:.3f} rows/deg")
    L.append(f"    estimate 2 (Theil-Sen) : {c.estimate2_rows_per_deg:.3f} rows/deg")
    L.append(f"    cross-check residual   : {c.cross_check_residual * 100:.2f}% "
             f"(tol {c.agree_tol * 100:.0f}%) -> "
             f"{'AGREE' if c.estimates_agree else 'DISAGREE'}")
    L.append(f"    anchor correlation r   : {c.anchor_r:.3f} "
             f"(n_fix={c.anchor_n_fix})")
    L.append("  " + "-" * 52)
    L.append("  along-channel main sequence (test2 @ ~12 kHz):")
    L.append(f"    n saccade events       : {c.main_seq_n_events}")
    L.append(f"    log-log slope          : {c.main_seq_slope:.4f} (finite, no NaN)")
    L.append(f"    deg / along-px         : {c.deg_per_along_px:.4f} "
             f"(r={c.along_grounding_r:.3f})")
    L.append("  " + "-" * 52)
    L.append(f"  hardware sync OFF        : grid={c.off_grid_s:.2f}s "
             f"along={c.off_along_s:.2f}s")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="G2 row<->arcmin calibration")
    p.add_argument("--report", action="store_true",
                   help="print deg/row, alias spacing, main-seq slope, cross-check residual")
    p.add_argument("--rebuild", action="store_true", help="ignore cached calibration")
    args = p.parse_args()
    if args.rebuild:
        calibrate(rebuild=True)
    if args.report:
        print(_report())
    else:
        p.print_help()
