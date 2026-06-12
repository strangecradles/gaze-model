"""aoslo_headtohead.py — head-to-head of OUR DPF vs the Azimipour-2018 strip
registration on the UC-Davis AO-SLO public dataset (Jonnal,
`intraframe_motion_correction`).

This is a self-contained ADAPTER + comparison driver. It does NOT route the
AO-SLO data through the khz2d column-raster harness; instead it wires the
axis-agnostic particle filter (filter.py / decoder.py / dynamics.py) to the
AO-SLO geometry and runs it next to a faithful py3 port of the reference
strip-registration method, on the SIMULATED set (which ships ground-truth eye
traces) and the REAL frames.

Dataset (verified empirically, see results/aoslo_headtohead.md):
  external/intraframe_motion_correction/
    object/full_mosaic.npy           512x512 simulated retinal cone mosaic (atlas)
    slo_frames_simulated/NNN.npy      200 frames, 128x128, 30 Hz, motion-affected
    slo_frames_simulated/resources/   eye_trace_x.npy, eye_trace_y.npy (200,128)
                                      = the ACTUAL ground-truth used to build the
                                      frames (1.5x the raw simulated_eye_traces/),
                                      and motion_free.npy (128x128) = mosaic[100:228,100:228]
    slo_frames_real_large/NNN.npy     100 real UC-Davis AO-SLO frames, 512x512

Geometry (verified to machine precision):
  Each acquired LINE is one HORIZONTAL row of the frame (fast scan = horizontal);
  the raster advances DOWN one row per line (slow scan = vertical). So:
      ALONG  = horizontal (x, mosaic columns)  -> trusted, densely sampled per line
      PERP   = vertical   (y, mosaic rows)     -> aliased, ONE row per time step
  For simulated frame f, line idx the sampled gaze is
      perp(absolute)  = 100 + idx + gy[f,idx]
      along(absolute) = 100      + gx[f,idx]
  and decoder.render(perp, along, full_mosaic, 128) reproduces the stored line
  to ~1e-15. We track the eye RESIDUAL (raster ramp removed) by feeding the
  filter a per-line atlas window centred on the nominal raster row.

Scale: 2 deg / 512 px = 0.00390625 deg/px = 0.234375 arcmin/px (mosaic/instrument
sampling; the 128-px simulated frame is a 0.5 deg window of the same mosaic).

Run:  python aoslo_headtohead.py            # full run (caches), writes results/
      python aoslo_headtohead.py --rebuild  # ignore caches
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import decoder
import dynamics
import calib
import filter as flt

# ---------------------------------------------------------------------------
# Paths & scale
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "external", "intraframe_motion_correction")
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(RESULTS, "aoslo_cache")
os.makedirs(CACHE, exist_ok=True)

DEG_PER_PX = 2.0 / 512.0
ARCMIN_PER_PX = DEG_PER_PX * 60.0          # 0.234375
FRAME_RATE_HZ = 30.0
N_LINES = 128
LINE_RATE_HZ = FRAME_RATE_HZ * N_LINES     # 3840 Hz
DT = 1.0 / LINE_RATE_HZ
MX0 = MY0 = 100                            # mosaic crop origin used by create_simulated_images.py


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_simulated():
    """Return (frames (F,128,128), gx (F,128), gy (F,128), mosaic (512,512))."""
    fl = sorted(glob.glob(os.path.join(EXT, "slo_frames_simulated", "*.npy")))
    fl = [f for f in fl if os.path.basename(os.path.dirname(f)) == "slo_frames_simulated"]
    frames = np.array([np.load(f) for f in fl], dtype=np.float64)
    res = os.path.join(EXT, "slo_frames_simulated", "resources")
    gx = np.load(os.path.join(res, "eye_trace_x.npy")).astype(np.float64)
    gy = np.load(os.path.join(res, "eye_trace_y.npy")).astype(np.float64)
    mosaic = np.load(os.path.join(EXT, "object", "full_mosaic.npy")).astype(np.float64)
    F = frames.shape[0]
    return frames, gx[:F], gy[:F], mosaic


def load_real(n=None):
    fl = sorted(glob.glob(os.path.join(EXT, "slo_frames_real_large", "*.npy")))
    if n is not None:
        fl = fl[:n]
    frames = np.array([np.load(f) for f in fl], dtype=np.float64)
    return frames


# ---------------------------------------------------------------------------
# SOTA baseline — Azimipour 2018 strip registration (faithful py3 port)
# ---------------------------------------------------------------------------
# Reimplemented in python3 from demonstrate_registration.py (original is py2).
# Equations 1-2 of Azimipour et al., PLoS One 2018: per-strip FFT2
# cross-correlation against a reference, with the horizontal-strip mean-bias
# removal the paper/script use in lieu of full normalisation.

def _parabolic(cm, c0, cp):
    """Sub-pixel peak offset from three samples (minus, peak, plus)."""
    den = (cm - 2.0 * c0 + cp)
    return 0.5 * (cm - cp) / den if abs(den) > 1e-12 else 0.0


def _strip_register_frame(target, f_ref_conj, ref_shape, strip_width, subpixel=True):
    """Per-row strip lags of `target` against a reference (its conj-FFT2 given).

    Returns (x_lag (R,), y_lag (R,), corr (R,)) per the reference script:
    a strip_width-row window centred on each row, FFT2 cross-correlated with the
    reference, mean-bias removed per output row, argmax -> circular-corrected lag.
    With ``subpixel`` a standard parabolic 3-point peak fit refines the integer
    lag (the fair best-case for a strip-registration method; the original script
    reports integer lags only)."""
    n_rows, n_cols = ref_shape
    rr = np.arange(n_rows)
    x_lag = np.zeros(n_rows); y_lag = np.zeros(n_rows); corr = np.zeros(n_rows)
    for r in range(n_rows):
        mask = (np.abs(rr - r) <= strip_width // 2).astype(np.float64)
        tar = target * mask[:, None]
        f_tar = np.conj(np.fft.fft2(tar))
        xcorr = np.abs(np.fft.ifft2(f_tar * f_ref_conj))
        xcorr = (xcorr.T - xcorr.mean(axis=1)).T           # horizontal-strip bias removal
        yi, xi = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        c = xcorr[yi, xi]
        dy = dx = 0.0
        if subpixel:
            dy = _parabolic(xcorr[(yi - 1) % n_rows, xi], c, xcorr[(yi + 1) % n_rows, xi])
            dx = _parabolic(xcorr[yi, (xi - 1) % n_cols], c, xcorr[yi, (xi + 1) % n_cols])
        yl = yi - n_rows if yi > n_rows // 2 else yi
        xl = xi - n_cols if xi > n_cols // 2 else xi
        x_lag[r] = xl + dx; y_lag[r] = yl + dy; corr[r] = c
    return x_lag, y_lag, corr


def sota_simulated(frames, mosaic, strip_width=13, rebuild=False):
    """Strip-register every simulated frame against the MOTION-FREE object so the
    recovered per-strip lag is an ABSOLUTE eye-position estimate (the regime in
    which the simulated GT lives). Returns (sx (F,128), sy (F,128), corr)."""
    cf = os.path.join(CACHE, f"sota_sim_sw{strip_width}.npz")
    if os.path.exists(cf) and not rebuild:
        d = np.load(cf)
        return d["sx"], d["sy"], d["corr"], d["sx_int"], d["sy_int"]
    motion_free = mosaic[MY0:MY0 + N_LINES, MX0:MX0 + N_LINES]
    # reference fft2 (conjugate of the reference, matching the script's f_ref usage)
    f_ref = np.fft.fft2(motion_free)
    F = frames.shape[0]
    sx = np.zeros((F, N_LINES)); sy = np.zeros((F, N_LINES)); cc = np.zeros((F, N_LINES))
    sxi = np.zeros((F, N_LINES)); syi = np.zeros((F, N_LINES))
    t0 = time.time()
    for f in range(F):
        xl, yl, c = _strip_register_frame(frames[f], f_ref, motion_free.shape,
                                          strip_width, subpixel=True)
        xi, yi, _ = _strip_register_frame(frames[f], f_ref, motion_free.shape,
                                          strip_width, subpixel=False)
        sx[f] = xl; sy[f] = yl; cc[f] = c; sxi[f] = xi; syi[f] = yi
        if f % 40 == 0:
            print(f"  [SOTA sim] frame {f}/{F} ({time.time()-t0:.0f}s)")
    np.savez(cf, sx=sx, sy=sy, corr=cc, sx_int=sxi, sy_int=syi)
    return sx, sy, cc, sxi, syi


# ---------------------------------------------------------------------------
# OUR DPF — adapter to the AO-SLO geometry
# ---------------------------------------------------------------------------

def configure_dynamics_for_aoslo():
    """Calibrate the IMM prior's units (atlas 'rows' == AO-SLO pixels) and timing.

    The dynamics constants ship calibrated to OUR instrument (124.6 rows/deg,
    0.481'/row). Here 1 px = 0.234375' (256 px/deg), so we rescale the angular
    constants into this dataset's pixel units. Saccade params are kept (the
    self-avoiding-walk GT has no real saccades, so the OU drift mode dominates);
    the OU stationary velocity is set to match the measured GT drift+tremor.
    """
    dynamics.ROWS_PER_DEG = 1.0 / DEG_PER_PX                 # 256 px/deg
    dynamics.ARCMIN_PER_ROW = ARCMIN_PER_PX                  # 0.234375 '/px
    dynamics.A0_ROWS = dynamics.A0_ARCMIN / ARCMIN_PER_PX    # main-seq knee in px
    dynamics.VMAX_ROWS_S = 826.0 / DEG_PER_PX               # keep 826 deg/s asymptote, in px/s
    # measured GT velocity std ~1275 px/s (drift+tremor); let the OU prior carry it
    dynamics.SIGMA_V_PURSUIT_ROWS_S = 1200.0
    dynamics.TAU_PURSUIT_S = 0.02
    dynamics.ACCEL_CAP_ROWS_S2 = 5.0e5
    # near-mask alias window for the point estimate: use the cone vertical period
    calib.ALIAS_SPACING_ROWS = 5.0


def _along_measure(obs, ref_row, nominal_col, search=8):
    """Trusted ALONG (horizontal) position via 1D normalised cross-correlation of
    the observed line against the nominal atlas row. Returns absolute mosaic col."""
    L = obs.shape[0]
    o = (obs - obs.mean()) / (obs.std() + 1e-9)
    best_c, best_lag = -np.inf, 0
    for lag in range(-search, search + 1):
        c0 = nominal_col + lag
        seg = ref_row[c0:c0 + L]
        if seg.shape[0] < L:
            continue
        s = (seg - seg.mean()) / (seg.std() + 1e-9)
        c = float((o * s).mean())
        if c > best_c:
            best_c, best_lag = c, lag
    return nominal_col + best_lag


def run_dpf_simulated(frames, mosaic, n_particles=250, pad=14, seed=0,
                      perp_spread=5.0, along_spread=3.0, rebuild=False):
    """Run the particle filter per frame over the 128 lines; return per-line
    recovered eye residuals est_gx (F,128), est_gy (F,128) and est NCC."""
    cf = os.path.join(CACHE, f"dpf_sim_n{n_particles}_p{pad}.npz")
    if os.path.exists(cf) and not rebuild:
        d = np.load(cf); return d["gx"], d["gy"], d["ncc"]
    configure_dynamics_for_aoslo()
    F = frames.shape[0]
    est_gx = np.zeros((F, N_LINES)); est_gy = np.zeros((F, N_LINES))
    est_ncc = np.zeros((F, N_LINES))
    t0 = time.time()
    for f in range(F):
        rng = np.random.default_rng(seed + f)
        # init residual cloud at nominal (eye~0), absolute along ~ MX0
        a0 = _along_measure(frames[f][0], mosaic[MY0], MX0)
        st = flt.init_filter(n_particles, pad, float(a0), perp_spread, along_spread, rng=rng)
        pf = None
        for idx in range(N_LINES):
            nominal_row = MY0 + idx
            win = mosaic[nominal_row - pad: nominal_row + pad + 1, :]   # (2pad+1, 512)
            obs = frames[f][idx]
            along_meas = _along_measure(obs, mosaic[nominal_row], MX0)
            if pf is None:
                pf = flt.ParticleFilter(st, win, N_LINES, col_step=1.0,
                                        sigma_along=2.0, reseed_perp_sigma=4.0,
                                        ncc_loss_window=8, couple_sigma=0.0,
                                        likelihood="physics")
            else:
                pf.atlas = win
            post = pf.step(obs, float(along_meas), DT, rng, coarse_anchor=float(pad))
            est_gy[f, idx] = post.est_perp - pad           # residual rows = gy
            est_gx[f, idx] = post.est_along - MX0          # absolute col - origin = gx
            est_ncc[f, idx] = post.max_ncc
        if f % 20 == 0:
            print(f"  [DPF sim] frame {f}/{F} ({time.time()-t0:.0f}s)")
    np.savez(cf, gx=est_gx, gy=est_gy, ncc=est_ncc)
    return est_gx, est_gy, est_ncc


# ---------------------------------------------------------------------------
# REAL frames (no GT): precision + qualitative agreement
# ---------------------------------------------------------------------------

def sota_real(frames, ref, strip_width=13):
    """Strip-register each real frame to a reference frame (frame 0). Returns
    sx,sy (F, R) per-row lags relative to the reference (sub-pixel)."""
    f_ref = np.fft.fft2(ref)
    F, R, _ = frames.shape
    sx = np.zeros((F, R)); sy = np.zeros((F, R))
    t0 = time.time()
    for f in range(F):
        xl, yl, _ = _strip_register_frame(frames[f], f_ref, ref.shape, strip_width,
                                          subpixel=True)
        sx[f] = xl; sy[f] = yl
        if f % 5 == 0:
            print(f"  [SOTA real] frame {f}/{F} ({time.time()-t0:.0f}s)")
    return sx, sy


def run_dpf_real(frames, ref, n_particles=200, pad=24, seed=0):
    """Run the PF per real frame against a reference frame as atlas. Returns
    est_x, est_y (F, R) recovered position relative to the reference."""
    configure_dynamics_for_aoslo()
    F, R, W = frames.shape
    ex = np.zeros((F, R)); ey = np.zeros((F, R))
    t0 = time.time()
    for f in range(F):
        rng = np.random.default_rng(seed + f)
        a0 = _along_measure(frames[f][pad], ref[pad], 0, search=pad)
        st = flt.init_filter(n_particles, pad, float(a0), 8.0, 5.0, rng=rng)
        pf = None
        for idx in range(R):
            nr = idx
            lo = max(0, nr - pad); hi = min(R, nr + pad + 1)
            win = ref[lo:hi, :]
            if win.shape[0] < 2 * pad + 1:        # edge pad
                padrows = np.zeros((2 * pad + 1 - win.shape[0], W))
                win = np.vstack([padrows, win] if lo == 0 else [win, padrows])
            obs = frames[f][idx]
            am = _along_measure(obs, ref[nr], 0, search=pad)
            if pf is None:
                pf = flt.ParticleFilter(st, win, W, col_step=1.0, sigma_along=2.0,
                                        reseed_perp_sigma=5.0, ncc_loss_window=10,
                                        couple_sigma=0.0, likelihood="physics")
            else:
                pf.atlas = win
            post = pf.step(obs, float(am), DT, rng, coarse_anchor=float(pad))
            ey[f, idx] = post.est_perp - pad
            ex[f, idx] = post.est_along
        if f % 5 == 0:
            print(f"  [DPF real] frame {f}/{F} ({time.time()-t0:.0f}s)")
    return ex, ey


def _jitter_arcmin(trace):
    """Estimation-noise proxy: median |2nd difference| per row (robust to smooth
    true drift), in arcmin. Lower = more precise."""
    d2 = np.diff(trace, n=2, axis=1)
    return float(np.nanmedian(np.abs(d2)) * ARCMIN_PER_PX)


def real_comparison(n_frames=12, rebuild=False):
    cf = os.path.join(CACHE, f"real_n{n_frames}.npz")
    if os.path.exists(cf) and not rebuild:
        d = np.load(cf)
        return {k: d[k] for k in d.files}
    frames = load_real(n=n_frames)
    ref = frames[0]
    print(f"  real frames {frames.shape}, reference = frame 0")
    sx, sy = sota_real(frames, ref)
    ex, ey = run_dpf_real(frames, ref)
    # align each method per-axis to remove constant offset, compare agreement.
    # The SOTA-lag vs DPF-position sign is set empirically (max correlation).
    def al(a):
        return a - np.nanmean(a)
    sgx = 1.0 if np.corrcoef(ex.ravel(), sx.ravel())[0, 1] >= 0 else -1.0
    sgy = 1.0 if np.corrcoef(ey.ravel(), sy.ravel())[0, 1] >= 0 else -1.0
    sx_s = sgx * sx
    sy_s = sgy * sy
    agree_x = float(np.sqrt(np.nanmean((al(ex) - al(sx_s)) ** 2)) * ARCMIN_PER_PX)
    agree_y = float(np.sqrt(np.nanmean((al(ey) - al(sy_s)) ** 2)) * ARCMIN_PER_PX)
    cx = float(np.corrcoef(ex.ravel(), sx_s.ravel())[0, 1])
    cy = float(np.corrcoef(ey.ravel(), sy_s.ravel())[0, 1])
    out = dict(sx=sx, sy=sy, ex=ex, ey=ey, sgx=sgx, sgy=sgy,
               sota_jitter_x=_jitter_arcmin(sx), sota_jitter_y=_jitter_arcmin(sy),
               dpf_jitter_x=_jitter_arcmin(ex), dpf_jitter_y=_jitter_arcmin(ey),
               agree_x=agree_x, agree_y=agree_y, corr_x=cx, corr_y=cy,
               n_frames=n_frames)
    np.savez(cf, **out)
    return out


def make_real_figure(real, path):
    sx, sy, ex, ey = real["sx"], real["sy"], real["ex"], real["ey"]
    sgx = float(real["sgx"]); sgy = float(real["sgy"])
    R = sx.shape[1]
    t = np.arange(R) * DT * 1e3
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.patch.set_facecolor("white")
    for col, (lab, sot, dp) in enumerate([
            ("horizontal (along / x)", sgx * sx[1], ex[1] - np.nanmean(ex[1])),
            ("vertical (perp / y)", sgy * sy[1], ey[1] - np.nanmean(ey[1]))]):
        a = ax[col]
        sot = sot - np.nanmean(sot)
        a.plot(t, sot * ARCMIN_PER_PX, color="#d1495b", lw=1.3, label="SOTA strip-reg")
        a.plot(t, dp * ARCMIN_PER_PX, color="#1b8a5a", lw=1.3, label="our DPF")
        a.set_title(f"REAL frame 1 vs ref: {lab} (no GT)", fontsize=11)
        a.set_xlabel("time within frame (ms)"); a.set_ylabel("recovered (arcmin)")
        a.grid(alpha=0.25)
        if col == 0:
            a.legend(fontsize=9)
    fig.suptitle("AO-SLO REAL frames — recovered trajectories (precision/agreement, "
                 "no ground truth)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rms_arcmin(est, gt):
    """RMS of (est - gt) in arcmin after removing a single constant offset per
    axis (the absolute origin of an eye trace is arbitrary / reference-relative)."""
    e = est - gt
    e = e - np.nanmean(e)
    return float(np.sqrt(np.nanmean(e ** 2)) * ARCMIN_PER_PX)


def accuracy_table(sx, sy, dpf_gx, dpf_gy, gx, gy):
    """Absolute accuracy (RMS arcmin vs GT) for both methods, per axis + 2D."""
    # SOTA lag sign: recovered lag ~ +gx,+gy (verified by correlation); flip if needed
    def _sign(a, b):
        return 1.0 if np.nansum(a * b) >= 0 else -1.0
    ssx, ssy = _sign(sx, gx), _sign(sy, gy)
    out = {}
    out["sota_x"] = _rms_arcmin(ssx * sx, gx)
    out["sota_y"] = _rms_arcmin(ssy * sy, gy)
    out["sota_2d"] = float(np.sqrt(out["sota_x"] ** 2 + out["sota_y"] ** 2))
    out["dpf_x"] = _rms_arcmin(dpf_gx, gx)
    out["dpf_y"] = _rms_arcmin(dpf_gy, gy)
    out["dpf_2d"] = float(np.sqrt(out["dpf_x"] ** 2 + out["dpf_y"] ** 2))
    out["_sota_sign"] = (ssx, ssy)
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(sx, sy, dpf_gx, dpf_gy, gx, gy, acc, path):
    ssx, ssy = acc["_sota_sign"]
    # align SOTA & DPF to GT by removing per-axis mean offset for plotting
    def al(a, g):
        return a - np.nanmean(a - g)
    f_show = 7
    t = np.arange(N_LINES) * DT * 1e3   # ms within a frame
    plt.style.use("default")
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("white")

    for col, (lab, gt, sot, dp) in enumerate([
        ("horizontal  (along / x)", gx[f_show], ssx * sx[f_show], dpf_gx[f_show]),
        ("vertical  (perp / y)", gy[f_show], ssy * sy[f_show], dpf_gy[f_show]),
    ]):
        a = ax[0, col]
        a.plot(t, gt * ARCMIN_PER_PX, "-", color="0.1", lw=3.0,
               label="ground truth", zorder=1, alpha=0.85)
        a.plot(t, al(sot, gt) * ARCMIN_PER_PX, color="#d1495b", lw=1.4,
               label="SOTA strip-reg", alpha=0.9, zorder=2)
        a.plot(t, al(dp, gt) * ARCMIN_PER_PX, color="#1b8a5a", lw=1.4,
               label="our DPF", alpha=0.95, zorder=3)
        a.set_title(f"frame {f_show}: {lab}", fontsize=12)
        a.set_xlabel("time within frame (ms)"); a.set_ylabel("eye position (arcmin)")
        a.grid(alpha=0.25)
        if col == 0:
            a.legend(fontsize=9, loc="best")

    # accuracy bars
    a = ax[1, 0]
    labels = ["x (along)", "y (perp)", "2D"]
    sota_vals = [acc["sota_x"], acc["sota_y"], acc["sota_2d"]]
    dpf_vals = [acc["dpf_x"], acc["dpf_y"], acc["dpf_2d"]]
    xp = np.arange(3); w = 0.36
    a.bar(xp - w / 2, sota_vals, w, color="#d1495b", label="SOTA strip-reg")
    a.bar(xp + w / 2, dpf_vals, w, color="#1b8a5a", label="our DPF")
    for i, v in enumerate(sota_vals):
        a.text(i - w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(dpf_vals):
        a.text(i + w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(xp); a.set_xticklabels(labels)
    a.set_ylabel("absolute RMS error (arcmin)")
    a.set_title("Absolute accuracy vs ground truth (all 200 sim frames)", fontsize=12)
    a.legend(fontsize=9); a.grid(alpha=0.25, axis="y")

    # scatter recovered-vs-GT (vertical / aliased axis = the interesting one)
    a = ax[1, 1]
    g = gy.ravel(); s = (ssy * sy).ravel(); dp = dpf_gy.ravel()
    s = s - np.nanmean(s - g); dp = dp - np.nanmean(dp - g)
    a.scatter(g * ARCMIN_PER_PX, s * ARCMIN_PER_PX, s=4, color="#d1495b",
              alpha=0.25, label="SOTA")
    a.scatter(g * ARCMIN_PER_PX, dp * ARCMIN_PER_PX, s=4, color="#1b8a5a",
              alpha=0.25, label="DPF")
    lim = np.nanpercentile(np.abs(g * ARCMIN_PER_PX), 99) * 1.2
    a.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.6)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel("GT vertical pos (arcmin)"); a.set_ylabel("recovered (arcmin)")
    a.set_title("Vertical (aliased) axis: recovered vs GT", fontsize=12)
    a.legend(fontsize=9); a.grid(alpha=0.25)

    fig.suptitle("AO-SLO head-to-head — our particle filter vs Azimipour-2018 "
                 "strip registration (UC-Davis simulated set, ground truth)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-particles", type=int, default=250)
    a = ap.parse_args()

    print("Loading simulated AO-SLO set ...")
    frames, gx, gy, mosaic = load_simulated()
    print(f"  frames {frames.shape}, GT trace {gx.shape}, mosaic {mosaic.shape}")

    print("Running SOTA strip registration (Azimipour 2018) ...")
    sx, sy, scorr, sxi, syi = sota_simulated(frames, mosaic, rebuild=a.rebuild)

    print("Running OUR DPF adapter ...")
    dpf_gx, dpf_gy, dpf_ncc = run_dpf_simulated(
        frames, mosaic, n_particles=a.n_particles, rebuild=a.rebuild)

    acc = accuracy_table(sx, sy, dpf_gx, dpf_gy, gx, gy)
    acc_int = accuracy_table(sxi, syi, dpf_gx, dpf_gy, gx, gy)
    acc["sota_int_x"] = acc_int["sota_x"]; acc["sota_int_y"] = acc_int["sota_y"]
    acc["sota_int_2d"] = acc_int["sota_2d"]
    print("\n=== ABSOLUTE ACCURACY (RMS arcmin vs GT, all 200 frames) ===")
    print(f"  SOTA (integer lag) x={acc['sota_int_x']:.3f} y={acc['sota_int_y']:.3f} 2D={acc['sota_int_2d']:.3f}")
    print(f"  SOTA (sub-pixel)   x={acc['sota_x']:.3f} y={acc['sota_y']:.3f} 2D={acc['sota_2d']:.3f}")
    print(f"  DPF                x={acc['dpf_x']:.3f} y={acc['dpf_y']:.3f} 2D={acc['dpf_2d']:.3f}")

    make_figure(sx, sy, dpf_gx, dpf_gy, gx, gy, acc,
                os.path.join(RESULTS, "aoslo_headtohead.png"))

    print("\nRunning REAL-frame comparison (no GT: precision + agreement) ...")
    real = real_comparison(n_frames=12, rebuild=a.rebuild)
    print(f"  jitter (median |2nd-diff|, arcmin):  SOTA x={real['sota_jitter_x']:.4f} "
          f"y={real['sota_jitter_y']:.4f}   DPF x={real['dpf_jitter_x']:.4f} "
          f"y={real['dpf_jitter_y']:.4f}")
    print(f"  inter-method agreement RMS arcmin:  x={real['agree_x']:.3f} "
          f"y={real['agree_y']:.3f}   corr x={real['corr_x']:.3f} y={real['corr_y']:.3f}")
    make_real_figure(real, os.path.join(RESULTS, "aoslo_headtohead_real.png"))

    # stash a small summary npz for the md writer / reproducibility
    np.savez(os.path.join(CACHE, "summary.npz"),
             **{k: v for k, v in acc.items() if not k.startswith("_")})
    return acc, real


if __name__ == "__main__":
    main()
