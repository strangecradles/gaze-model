"""dewarp_diag.py — Tasks 1 & 2 of the intra-frame-dewarp investigation.

NO filter change. Works purely from the committed line cache (lm["rdx"], the
per-line horizontal NCC argmax). Emits:
  results/dewarp_baseline.json      (Task 1: reproduce 4.33' rdx_scatter)
  results/dewarp_regressions.json   (Task 2: 3 diagnostic regressions)

The primary metric (identical to step0_recalib.real_line_scatter_px):
  residual = rdx - gaussian(median(rdx,7), 10ms)   [masked to FOV]
  rdx_scatter = std(residual) * ARC_PER_PX   [arcmin]

Regressors (each -> R^2 + slope on the masked residual, in px):
  (a) within-frame raster position  = col (slow galvo axis, 0..W-1)   H1: >0
  (b) per-frame gaze displacement   = |chain inc| of the frame        H1: scales
  (c) CONTROL fast-axis position    = lam_v (vertical/resonant)        flat if scanner exonerated
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import khz2d
import people_fov_pf as pf

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX


def linfit(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS y~x; return slope, intercept, R^2, n, pearson r."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    g = np.isfinite(x) & np.isfinite(y)
    x = x[g]; y = y[g]
    n = x.size
    if n < 3 or x.std() < 1e-12:
        return dict(slope=0.0, intercept=float(np.mean(y)) if n else 0.0,
                    r2=0.0, r=0.0, n=int(n))
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = slope * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(slope=float(slope), intercept=float(b), r2=float(r2), r=r, n=int(n))


def residual_px(lm: dict, m: np.ndarray) -> np.ndarray:
    """rdx minus 10ms-smoothed track (px), NaN outside mask."""
    rdx = lm["rdx"].astype(float)
    fs = float(lm["line_rate"])
    filled = khz2d.fill_nan(np.where(m, rdx, np.nan))
    sm = gaussian_filter1d(median_filter(filled, 7), fs * 0.01)
    return np.where(m, rdx - sm, np.nan)


def main(person: str = "Igor"):
    sub = pf.subject_by_name(person)
    lm = pf.build_line_measurements(sub)          # cache; no rebuild
    ch = pf.build_chain(sub)                       # for per-frame inc
    m = pf.fov_mask(lm)
    fs = float(lm["line_rate"])

    resid = residual_px(lm, m)                     # px, NaN off-mask
    rscat_px = float(np.nanstd(resid[m]))
    rscat_arc = rscat_px * ARC
    n_lines = int(m.sum())

    os.makedirs(RESULTS, exist_ok=True)
    base = dict(person=person, capture=sub.stem, stimulus=pf.STIMULUS,
                rdx_scatter_arcmin=rscat_arc, rdx_scatter_px=rscat_px,
                arc_per_px=ARC, n_lines=n_lines,
                n_frames=int(lm["frame"].max()) + 1,
                line_rate_hz=fs, fps=float(lm["fps"]),
                note="residual = rdx - 10ms gaussian(median7); std over FOV mask")
    with open(os.path.join(RESULTS, "dewarp_baseline.json"), "w") as f:
        json.dump(base, f, indent=2)
    print(f"[Task1] {person}: rdx_scatter = {rscat_arc:.3f}' "
          f"({rscat_px:.3f}px, n={n_lines}) -> dewarp_baseline.json")

    # ---- Task 2 regressions (masked lines only) ----
    mi = np.where(m)[0]
    res_m = resid[mi]
    col_m = lm["col"][mi].astype(float)
    lam_m = lm["lam_v"][mi].astype(float)
    fr_m = lm["frame"][mi].astype(int)

    # (a) pooled residual ~ col
    a_pool = linfit(col_m, res_m)
    # (a') per-frame-detrended: regress residual ~ (col within frame), removing
    #      per-frame residual mean so a *common* intra-frame ramp shows up even if
    #      its sign were constant. col already identical range per frame.
    fmean = {}
    for fnum in np.unique(fr_m):
        sel = fr_m == fnum
        fmean[fnum] = res_m[sel].mean()
    res_dt = res_m - np.array([fmean[f] for f in fr_m])
    a_detr = linfit(col_m, res_dt)
    # (a'') mean within-frame slope: per-frame OLS slope of residual vs col, then
    #       summarize. A real intra-frame distortion gives a consistent sign.
    slopes = []
    for fnum in np.unique(fr_m):
        sel = fr_m == fnum
        if sel.sum() >= 20:
            ft = linfit(col_m[sel], res_m[sel])
            slopes.append(ft["slope"])
    slopes = np.array(slopes)
    a_within = dict(mean_slope=float(np.mean(slopes)),
                    median_slope=float(np.median(slopes)),
                    std_slope=float(np.std(slopes)),
                    frac_positive=float(np.mean(slopes > 0)),
                    n_frames=int(slopes.size),
                    note="per-frame OLS slope of residual(px) vs col; H1 -> consistent sign")
    # residual mean/std binned by col (reveals frame-edge structure)
    nb = 16
    edges = np.linspace(0, lm["col"].max() + 1, nb + 1)
    bidx = np.clip(np.digitize(col_m, edges) - 1, 0, nb - 1)
    col_bins = []
    for bb in range(nb):
        s = bidx == bb
        if s.any():
            col_bins.append(dict(col_lo=float(edges[bb]), col_hi=float(edges[bb + 1]),
                                 resid_mean_px=float(res_m[s].mean()),
                                 resid_std_arcmin=float(res_m[s].std() * ARC),
                                 n=int(s.sum())))

    # (b) per-frame gaze displacement |inc| vs within-frame residual scatter
    inc_x = ch["inc_x"]; inc_y = ch["inc_y"]
    inc_mag = np.sqrt(inc_x ** 2 + inc_y ** 2)          # px/frame, per chain frame
    fr_disp = []; fr_std = []; fr_absres = []
    for fnum in np.unique(fr_m):
        sel = fr_m == fnum
        if sel.sum() >= 20 and fnum < inc_mag.size:
            fr_disp.append(inc_mag[fnum])
            fr_std.append(res_m[sel].std())
            fr_absres.append(np.mean(np.abs(res_m[sel])))
    fr_disp = np.array(fr_disp); fr_std = np.array(fr_std); fr_absres = np.array(fr_absres)
    b_std = linfit(fr_disp, fr_std)        # per-frame residual std vs displacement
    b_abs = linfit(fr_disp, fr_absres)
    # per-line |resid| vs frame displacement (assign each line its frame's |inc|)
    line_disp = np.where(fr_m < inc_mag.size, inc_mag[np.clip(fr_m, 0, inc_mag.size - 1)], np.nan)
    b_line = linfit(line_disp, np.abs(res_m))

    # (c) CONTROL residual ~ fast-axis (vertical) POSITION lam_v.
    # The scanner test is position-dependence: a resonant-scanner artifact would make
    # the *signed* residual depend on fast-axis position. This must be flat.
    c_pool = linfit(lam_m, res_m)
    # residual MEAN binned by lam_v position (a position-bias scan, scanner test)
    lam_edges = np.quantile(lam_m, np.linspace(0, 1, 13))
    lam_edges = np.unique(lam_edges)
    lbidx = np.clip(np.digitize(lam_m, lam_edges) - 1, 0, len(lam_edges) - 2)
    lam_bins = []
    for bb in range(len(lam_edges) - 1):
        s = lbidx == bb
        if s.sum() > 50:
            lam_bins.append(dict(lam_lo=float(lam_edges[bb]), lam_hi=float(lam_edges[bb + 1]),
                                 resid_mean_px=float(res_m[s].mean()),
                                 resid_std_arcmin=float(res_m[s].std() * ARC), n=int(s.sum())))
    pos_bias_ptp_px = float(max(b["resid_mean_px"] for b in lam_bins)
                            - min(b["resid_mean_px"] for b in lam_bins)) if lam_bins else 0.0
    # SEPARATE diagnostic (NOT the scanner gate): |resid| vs |lam_v| is a motion-AMPLITUDE
    # coupling — large vertical motion -> large scatter on both axes. This supports H1
    # (motion-driven intra-frame distortion), it is not a scanner position artifact.
    c_abs = linfit(np.abs(lam_m), np.abs(res_m))

    regs = dict(
        person=person, capture=sub.stem,
        units=dict(residual="raster px", arc_per_px=ARC,
                   col="slow galvo axis index 0..W-1",
                   inc_mag="px per frame (chain increment magnitude)",
                   lam_v="vertical/fast-axis localization px"),
        a_within_frame_raster_pos=dict(
            pooled=a_pool, per_frame_detrended=a_detr,
            within_frame_slopes=a_within, col_bins=col_bins,
            H1_prediction="R2>0 / consistent within-frame slope"),
        b_per_frame_displacement=dict(
            perframe_std_vs_disp=b_std, perframe_absres_vs_disp=b_abs,
            perline_absres_vs_disp=b_line,
            H1_prediction="positive slope (scatter scales with frame motion)"),
        c_control_fast_axis=dict(
            resid_vs_lam_v_position=c_pool, lam_position_bins=lam_bins,
            position_bias_ptp_px=pos_bias_ptp_px,
            motion_coupling_absresid_vs_abslam=c_abs,
            requirement="signed residual vs fast-axis POSITION must be flat (R2~0)",
            FLAT=bool(c_pool["r2"] < 0.01),
            note=("FLAT gate uses signed position-dependence only. "
                  "motion_coupling_* (|resid| vs |lam|) is an H1-supporting amplitude "
                  "coupling, NOT a scanner artifact.")),
    )
    with open(os.path.join(RESULTS, "dewarp_regressions.json"), "w") as f:
        json.dump(regs, f, indent=2)

    print(f"[Task2a] residual~col   pooled R2={a_pool['r2']:.4f} slope={a_pool['slope']:.2e} | "
          f"detrended R2={a_detr['r2']:.4f} | within-frame mean_slope={a_within['mean_slope']:.2e} "
          f"frac+={a_within['frac_positive']:.2f}")
    print(f"[Task2b] |resid|~|inc| perframe-std R2={b_std['r2']:.4f} slope={b_std['slope']:.3f} | "
          f"perline R2={b_line['r2']:.4f} slope={b_line['slope']:.3f}")
    print(f"[Task2c] CONTROL(position) resid~lam_v R2={c_pool['r2']:.4f} slope={c_pool['slope']:.2e} "
          f"pos_bias_ptp={pos_bias_ptp_px:.3f}px  ->  FLAT={regs['c_control_fast_axis']['FLAT']}")
    print(f"         (motion-coupling |resid|~|lam| R2={c_abs['r2']:.4f} -> H1-supporting, not scanner)")
    print("-> results/dewarp_regressions.json")
    if not regs["c_control_fast_axis"]["FLAT"]:
        print("\n*** STOP CONDITION (Task 2c): position control NOT flat — scanner assumption suspect. ***")
    return base, regs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Igor")
    main(ap.parse_args().person)
