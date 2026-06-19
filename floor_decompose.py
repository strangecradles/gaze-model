"""floor_decompose.py — Phase 1 Tasks 2 & 3: C2 (reference-common systematic) and
C3 (real-motion UPPER BOUND), per subject. Consumes the multi-reference common track
from floor_multiref. PER SUBJECT only; never pooled.

C2 (Task 2): residual of the cross-reference COMMON track regressed against
reference-INDEPENDENT, content-dependent signatures — NCC peak height (qh), peak
curvature (parabolic sharpness), and sub-pixel phase frac(rdx). R²>0 => an argmax /
sub-pixel localization bias that does NOT depend on which reference is used (suspect 3).
Emit results/floor_c2.json.

C3 (Task 3): coherence of the common track under NON-adjacent splits (block-interleaved,
K=8/32/128 lines) — explicitly an UPPER BOUND on real motion (reference-common geometric
C2 still masquerades as coherent). Compared to the invalid even/odd crossover, plus a
low-frequency cross-check of the common track vs the machine tracker. Emit results/floor_c3.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import coherence

import khz2d
import people_fov_pf as pf
import floor_multiref as fm

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX


def _linfit(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    if x.size < 50 or x.std() < 1e-12:
        return dict(r2=0.0, slope=0.0, n=int(x.size))
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = s * x + b
    ss = np.sum((y - yh) ** 2); st = np.sum((y - y.mean()) ** 2)
    return dict(r2=float(1 - ss / st) if st > 0 else 0.0, slope=float(s), n=int(x.size))


def _multi_r2(X, y):
    """R^2 of OLS y ~ [X, 1] (X columns already feature-wise finite-filtered upstream)."""
    g = np.isfinite(y) & np.all(np.isfinite(X), 1)
    X, y = X[g], y[g]
    if y.size < 100:
        return 0.0
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ coef
    ss = np.sum((y - yh) ** 2); st = np.sum((y - y.mean()) ** 2)
    return float(1 - ss / st) if st > 0 else 0.0


def common_track(person, K=3):
    """Common track (px, current-frame coords) + validity + aligned committed lm."""
    sub = pf.subject_by_name(person)
    mr = fm.build_multiref(sub, K=K)
    lm = pf.build_line_measurements(sub)            # committed, same grid
    assert mr["rdx1"].shape == lm["rdx"].shape, "multiref/committed grid mismatch"
    R = np.stack([mr[f"rdx{j+1}"].astype(float) for j in range(K)], 0)
    Q = np.stack([mr[f"qh{j+1}"].astype(float) for j in range(K)], 0)
    con = mr["con"]; con_med = np.median(con)
    valid = (Q > pf.Q_FOV).all(0) & (con > pf.CONTRAST_FRAC * con_med) & np.isfinite(R).all(0)
    common = R.mean(0)                              # px
    return sub, mr, lm, common, valid, Q


def residual_common(common, valid, fs):
    filled = khz2d.fill_nan(np.where(valid, common, np.nan))
    sm = gaussian_filter1d(median_filter(filled, 7), fs * 0.01)
    return np.where(valid, common - sm, np.nan), sm


def task2_c2(person, K=3) -> dict:
    sub, mr, lm, common, valid, Q = common_track(person, K)
    fs = float(mr["line_rate"])
    resid, _ = residual_common(common, valid, fs)
    # reference-independent, content-dependent features
    qh_mean = Q.mean(0)                              # NCC peak height (mean across refs)
    prof = lm["prof"].astype(np.float32)            # (N, 2*PADH+1) horizontal NCC profile
    kpk = np.argmax(prof, 1)
    rows = np.arange(prof.shape[0])
    kc = np.clip(kpk, 1, prof.shape[1] - 2)
    curvature = (2 * prof[rows, kc] - prof[rows, kc - 1] - prof[rows, kc + 1])  # sharpness
    subphase = common - np.round(common)            # sub-pixel phase in [-0.5,0.5]
    subphase_bias = np.abs(subphase)                # parabolic bias peaks at |phase|->0.5
    m = valid & np.isfinite(resid)
    feats = dict(qh=_linfit(qh_mean[m], resid[m]),
                 curvature=_linfit(curvature[m], resid[m]),
                 subphase_abs=_linfit(subphase_bias[m], np.abs(resid[m])))
    Xall = np.stack([qh_mean[m], curvature[m], subphase_bias[m]], 1)
    r2_joint = _multi_r2(Xall, resid[m])
    c2_arc = float(np.sqrt(max(r2_joint, 0.0)) * np.nanstd(resid[m]) * ARC)
    out = dict(person=person, K=K, n=int(m.sum()),
               resid_common_arcmin=float(np.nanstd(resid[m]) * ARC),
               feature_r2=feats, joint_r2=r2_joint,
               C2_arcmin_lowerbound=c2_arc,
               note=("C2 = feature-explained part of the common-track residual. Features are "
                     "reference-independent (qh, peak curvature, sub-pixel phase) so this "
                     "isolates argmax/sub-pixel bias from real motion. LOWER bound on C2."))
    print(f"[C2 {person}] resid_common={out['resid_common_arcmin']:.3f}' joint_R2={r2_joint:.3f} "
          f"(qh {feats['qh']['r2']:.3f}, curv {feats['curvature']['r2']:.3f}, "
          f"phase {feats['subphase_abs']['r2']:.3f}) -> C2>={c2_arc:.3f}'")
    return out


def _crossover(tA, xA, tB, xB, fs, hp_frac=0.05, thresh=0.5, full=False):
    Rc = fs / 2.0
    t0 = max(tA.min(), tB.min()); t1 = min(tA.max(), tB.max())
    grid = np.arange(t0, t1, 1.0 / Rc)
    if grid.size < 2000:
        return (np.nan if not full else dict(crossover_hz=np.nan))
    ga = np.interp(grid, tA, xA); gb = np.interp(grid, tB, xB)
    ga = ga - gaussian_filter1d(ga, Rc * hp_frac); gb = gb - gaussian_filter1d(gb, Rc * hp_frac)
    nper = min(8192, grid.size // 8 * 2)
    f, C = coherence(ga, gb, fs=Rc, nperseg=nper)
    Cs = gaussian_filter1d(C, 2)
    below = Cs < thresh
    idx = np.where(below[1:] & (~below[:-1]))[0]
    cross = float(f[idx[0] + 1]) if idx.size else (float(f[-1]) if np.all(~below) else np.nan)
    if not full:
        return cross
    return dict(crossover_hz=cross,
                coh_0_10hz=float(np.nanmean(Cs[f < 10])),
                coh_10_50hz=float(np.nanmean(Cs[(f >= 10) & (f < 50)])),
                coh_50_200hz=float(np.nanmean(Cs[(f >= 50) & (f < 200)])))


def _grid_series(t, x, m, fs):
    Rc = fs / 2.0
    tv, xv = t[m], x[m]
    grid = np.arange(tv.min(), tv.max(), 1.0 / Rc)
    return grid, np.interp(grid, tv, xv)


def crossref_coherence(t, rdx_a, rdx_b, m, fs, thresh=0.5):
    """Coherence between two SAME-LINE localizations against DIFFERENT references.
    This is the properly non-adjacent split: shared content = real motion + reference-
    common systematic C2; decoherence = reference-specific C1. No even/odd adjacency."""
    return _crossover(t[m], rdx_a[m] * ARC, t[m], rdx_b[m] * ARC, fs, full=True)


def task3_c3(person, K=3, blocks=(8, 32, 128)) -> dict:
    sub, mr, lm, common, valid, Q = common_track(person, K)
    fs = float(mr["line_rate"]); ARCt = ARC
    t = mr["t"].astype(float); col = mr["col"]
    x = common * ARCt
    m = valid & np.isfinite(x)
    # even/odd (adjacency-confounded reference) for comparison
    ev = m & (col % 2 == 0); od = m & (col % 2 == 1)
    eo = _crossover(t[ev], x[ev], t[od], x[od], fs)
    # non-adjacent block-interleaved splits
    line_idx = np.arange(t.size)
    blk = {}
    for Kb in blocks:
        parity = (line_idx // Kb) % 2
        a = m & (parity == 0); b = m & (parity == 1)
        blk[f"block{Kb}"] = _crossover(t[a], x[a], t[b], x[b], fs)
    # PRIMARY non-adjacent split: cross-reference coherence (same line, different refs)
    rdx1 = mr["rdx1"].astype(float); rdx2 = mr["rdx2"].astype(float); rdx3 = mr["rdx3"].astype(float)
    cr = dict(rdx1_vs_rdx2=crossref_coherence(t, rdx1, rdx2, m, fs),
              rdx1_vs_rdx3=crossref_coherence(t, rdx1, rdx3, m, fs))
    # low-frequency cross-check of common track vs machine tracker
    refs = pf.compute_refs(sub, lm)
    sig = gaussian_filter1d(median_filter(khz2d.fill_nan(np.where(m, common, np.nan)), 7), fs * 0.01)
    trk = np.interp(t, refs["trk_t"] + refs["off"], refs["trk_x"])
    r_trk = float(khz2d.corr(sig[m], trk[m]))
    out = dict(person=person, K=K,
               crossref_coherence=cr,
               evenodd_crossover_hz=eo, block_crossover_hz=blk,
               common_vs_tracker_r=r_trk,
               note=("UPPER BOUND on real motion: reference-common geometric systematic (C2) "
                     "still masquerades as coherent. Block-interleaved splits are temporally "
                     "non-adjacent (unlike even/odd ~85us). If block crossover << even/odd, the "
                     "even/odd number was inflated by adjacency-correlated systematic."))
    print(f"[C3 {person}] crossref r1v2 cross={cr['rdx1_vs_rdx2']['crossover_hz']:.0f}Hz "
          f"coh(0-10/10-50/50-200)={cr['rdx1_vs_rdx2']['coh_0_10hz']:.2f}/"
          f"{cr['rdx1_vs_rdx2']['coh_10_50hz']:.2f}/{cr['rdx1_vs_rdx2']['coh_50_200hz']:.2f} | "
          f"evenodd={eo:.0f}Hz r_vs_tracker={r_trk:.3f}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()
    c2 = {}; c3 = {}
    for p in args.people:
        c2[p] = task2_c2(p, args.K)
        c3[p] = task3_c3(p, args.K)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(c2, open(os.path.join(RESULTS, "floor_c2.json"), "w"), indent=2)
    json.dump(c3, open(os.path.join(RESULTS, "floor_c3.json"), "w"), indent=2)
    print("-> results/floor_c2.json results/floor_c3.json")


if __name__ == "__main__":
    main()
